#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

constexpr int BOARD_SIZE = 15;
constexpr int BOARD_CELLS = BOARD_SIZE * BOARD_SIZE;
constexpr int EMPTY = 0;
constexpr int BLACK = 1;
constexpr int WHITE = 2;
constexpr int DRAW = 0;
constexpr int NO_WINNER = -1;
constexpr int SEP_TOKEN_ID = 228;
constexpr int MOVE_ID_OFFSET = 3;
constexpr int DEFAULT_SIMULATIONS = 1000;
constexpr int DEFAULT_CANDIDATE_LIMIT = 16;
constexpr int DEFAULT_ROLLOUT_LIMIT = 16;
constexpr double DEFAULT_EXPLORATION = 1.4;

// 要素数225の整数配列にboardという名前をつけている。usingは別名定義(型エイリアス)
//std::arrayは固定長配列の定義。
//<要素の型, 要素数>
using Board = std::array<int, BOARD_CELLS>; 

// 方向を表すタプルのリスト。固定値。
const std::array<std::pair<int, int>, 4> DIRECTIONS = {
    std::make_pair(1, 0),
    std::make_pair(0, 1),
    std::make_pair(1, 1),
    std::make_pair(1, -1),
};

// 構造体。これを定義しておくと、引数に構造体を渡すだけで終わり。
struct Options {
    std::uint64_t games = 0; // 実行するゲーム数
    int simulations = DEFAULT_SIMULATIONS; // 1手当たりのmctsシミュレーション回数
    int parallel = 1; // 並列実行するスレッド数 
    int candidate_limit = DEFAULT_CANDIDATE_LIMIT; // 探索候補手の上限数。上位何個を調べるか
    int rollout_limit = DEFAULT_ROLLOUT_LIMIT; // ロールアウトの最大手数
    double exploration = DEFAULT_EXPLORATION; // UCTの探索係数C。この値を大きくすると、まだあまり探索していない未踏の手を優先的に調べようとする。
    bool trace_plies = false; // 手番ごとの進捗ログを出力するかどうか。これをtrueにしておくと、デバックに有効
    bool has_seed = false; // 乱数シード値がコマンドラインから手動で指定されたかどうかのフラグ。falseならランダム。
    std::uint64_t seed = 0;

    // EVAL parameters
    bool is_eval = false; // trueなら、与えられた局面と手番に関する期待勝率を一瞬で計算する。
    std::string eval_board_str = ""; // 評価したい盤面状態を表す。
    int eval_move = -1; // 次の一手のインデックス。
};

// 構造体。対局シミュレーションが一局終了した時に、その結果を返す。
struct GameResult {
    int winner = DRAW;
    int plies = 0;
    bool foul_loss = false;
    std::string csv_rows;
};

// 引数なしでintを返す。天元を返す。
int center_index() {
    return (BOARD_SIZE / 2) * BOARD_SIZE + (BOARD_SIZE / 2);
}

// 引数として行と列を受け取り、整数を返す。二次元座標を一次元配列のインデックスに変換する。
int rc_to_idx(int row, int col) {
    return row * BOARD_SIZE + col;
}

// インデックスを2次元の座標に変換する
std::pair<int, int> idx_to_rc(int index) {
    return {index / BOARD_SIZE, index % BOARD_SIZE};
}

// 座標が盤面に収まっているか判定し、boolを返す。
bool inside(int row, int col) {
    return 0 <= row && row < BOARD_SIZE && 0 <= col && col < BOARD_SIZE;
}

// プレイヤー番号を入れると、相手プレイヤー番号を出力する
int other_player(int player) {
    return player == BLACK ? WHITE : BLACK;
}

// 盤面に空きマスが一つも残っていないかを判定。残っていなければtrue
bool board_is_full(const Board& board) {
    return std::none_of(board.begin(), board.end(), [](int cell) { return cell == EMPTY; });
}

// 盤面の石の数を数得て出力する。黒石と白石の数をそれぞれ。
std::pair<int, int> stone_counts(const Board& board) {
    int black_count = 0;
    int white_count = 0;
    for (int cell : board) {
        if (cell == BLACK) {
            ++black_count;
        } else if (cell == WHITE) {
            ++white_count;
        }
    }
    return {black_count, white_count};
}

// 元の盤面を壊さず、指定の位置に石を置いた時の一手先の未来の盤面を返す
Board board_with_move(const Board& board, int index, int player) {
    Board next_board = board;
    next_board[static_cast<std::size_t>(index)] = player;
    return next_board;
}

// 指定したマスに石を置いたとき、特定の方向に同じプレイヤーの石が何個連続して並んでいるかをカウントして返す
int contiguous_count(const Board& board, int index, int player, int dr, int dc) {
    int total = 1;
    auto [row, col] = idx_to_rc(index);

    for (int step = 1; inside(row + dr * step, col + dc * step); ++step) {
        if (board[static_cast<std::size_t>(rc_to_idx(row + dr * step, col + dc * step))] != player) {
            break;
        }
        ++total;
    }

    for (int step = 1; inside(row - dr * step, col - dc * step); ++step) {
        if (board[static_cast<std::size_t>(rc_to_idx(row - dr * step, col - dc * step))] != player) {
            break;
        }
        ++total;
    }

    return total;
}

// 指定したマスに石をおいたとき、どこかの方向で5個以上連続して並ぶかを判定する
bool has_five_or_more(const Board& board, int index, int player) {
    for (const auto& [dr, dc] : DIRECTIONS) {
        if (contiguous_count(board, index, player, dr, dc) >= 5) {
            return true;
        }
    }
    return false;
}

// 指定したマスに石をおいたとき、どこかの方向で6個以上連続して並ぶかを判定する
bool is_overline(const Board& board, int index, int player) {
    for (const auto& [dr, dc] : DIRECTIONS) {
        if (contiguous_count(board, index, player, dr, dc) >= 6) {
            return true;
        }
    }
    return false;
}

// 盤面全体をスキャンし、指定したプレイヤーが既に五連以上を達成しているかを判定する
bool player_has_five(const Board& board, int player) {
    for (int index = 0; index < BOARD_CELLS; ++index) {
        if (board[static_cast<std::size_t>(index)] == player && has_five_or_more(board, index, player)) {
            return true;
        }
    }
    return false;
}

// 盤面全体をスキャンし、指定したプレイヤーが既に長連を達成しているかを判定する
bool player_has_overline(const Board& board, int player) {
    for (int index = 0; index < BOARD_CELLS; ++index) {
        if (board[static_cast<std::size_t>(index)] == player && is_overline(board, index, player)) {
            return true;
        }
    }
    return false;
}

// 盤面全体を走査して、既に勝敗が決しているかを判定
int board_winner(const Board& board) {
    if (player_has_overline(board, BLACK)) {
        return WHITE;
    }
    if (player_has_five(board, BLACK)) {
        return BLACK;
    }
    if (player_has_five(board, WHITE)) {
        return WHITE;
    }
    return NO_WINNER;
}

// 指定したマスを通り、特定の方向に延びる直線状の全マスのインデックスを端から端まで集めてリストにする
std::vector<int> line_points_through(int index, int dr, int dc) {
    auto [row, col] = idx_to_rc(index);
    while (inside(row - dr, col - dc)) {
        row -= dr;
        col -= dc;
    }

    std::vector<int> points;
    while (inside(row, col)) {
        points.push_back(rc_to_idx(row, col));
        row += dr;
        col += dc;
    }
    return points;
}

// 指定した直線状に、そこにおけば即勝利になる空きマスがあるか探して、そのリストを返す
std::vector<int> immediate_wins_in_direction(
    const Board& board,
    int player,
    const std::vector<int>& line_points
) {
    std::vector<int> wins;
    wins.reserve(line_points.size());

    for (int candidate : line_points) {
        if (board[static_cast<std::size_t>(candidate)] != EMPTY) {
            continue;
        }
        Board next_board = board_with_move(board, candidate, player);
        if (player == BLACK && is_overline(next_board, candidate, BLACK)) {
            continue;
        }
        if (has_five_or_more(next_board, candidate, player)) {
            wins.push_back(candidate);
        }
    }

    return wins;
}

// 指定したマスに石を置いたとき、何方向に四連ができるかをカウントする
int count_four_directions(const Board& board, int move, int player) {
    int count = 0;
    for (const auto& [dr, dc] : DIRECTIONS) {
        const std::vector<int> line_points = line_points_through(move, dr, dc);
        if (!immediate_wins_in_direction(board, player, line_points).empty()) {
            ++count;
        }
    }
    return count;
}

// 指定したマスに石を置いたとき、何方向で活三が出来るかをカウントする
int count_open_three_directions(const Board& board, int move, int player) {
    int count = 0;
    for (const auto& [dr, dc] : DIRECTIONS) {
        const std::vector<int> line_points = line_points_through(move, dr, dc);
        bool found_open_three = false;
        for (int candidate : line_points) {
            if (board[static_cast<std::size_t>(candidate)] != EMPTY) {
                continue;
            }
            Board next_board = board_with_move(board, candidate, player);
            if (player == BLACK && is_overline(next_board, candidate, BLACK)) {
                continue;
            }
            const std::vector<int> winning_points =
                immediate_wins_in_direction(next_board, player, line_points);
            if (winning_points.size() >= 2) {
                found_open_three = true;
                break;
            }
        }
        if (found_open_three) {
            ++count;
        }
    }
    return count;
}

// 指定したマスに黒石を置いたとき、黒の禁じ手になるかどうかを判定する
bool is_forbidden_for_black(const Board& board, int index) {
    if (board[static_cast<std::size_t>(index)] != EMPTY) {
        return true;
    }

    const auto [black_count, white_count] = stone_counts(board);
    const int move_number = black_count + white_count;
    if (move_number == 0) {
        return index != center_index();
    }

    Board next_board = board_with_move(board, index, BLACK);
    if (is_overline(next_board, index, BLACK)) {
        return true;
    }
    if (count_four_directions(next_board, index, BLACK) >= 2) {
        return true;
    }
    if (count_open_three_directions(next_board, index, BLACK) >= 2) {
        return true;
    }
    return false;
}

// 石を置いた直後に、そのプレイヤーの勝利が決まるかを判定する
int winner_after_move(const Board& board, int index, int player) {
    if (player == BLACK && is_overline(board, index, BLACK)) {
        return WHITE;
    }
    if (has_five_or_more(board, index, player)) {
        return player;
    }
    return NO_WINNER;
}

// 指定マスと盤面の中央との距離の二乗を算出する
int center_distance_sq(int index) {
    auto [row, col] = idx_to_rc(index);
    const int center = BOARD_SIZE / 2;
    const int dr = row - center;
    const int dc = col - center;
    return dr * dr + dc * dc;
}

// 指定したマスの隣接する8マスに、既にどれだけ石が置かれているかをカウントする
int local_density(const Board& board, int index) {
    auto [row, col] = idx_to_rc(index);
    int score = 0;
    for (int dr = -1; dr <= 1; ++dr) {
        for (int dc = -1; dc <= 1; ++dc) {
            if (dr == 0 && dc == 0) {
                continue;
            }
            const int nr = row + dr;
            const int nc = col + dc;
            if (inside(nr, nc) && board[static_cast<std::size_t>(rc_to_idx(nr, nc))] != EMPTY) {
                ++score;
            }
        }
    }
    return score;
}

// あるマスに石を置いたとき、その手の形がどれくらい協力かを数値化する
double move_shape_score(const Board& board, int move, int player) {
    Board next_board = board_with_move(board, move, player);
    int longest = 0;
    int pressure = 0;
    for (const auto& [dr, dc] : DIRECTIONS) {
        const int length = contiguous_count(next_board, move, player, dr, dc);
        longest = std::max(longest, length);
        pressure += length * length;
    }

    double score = static_cast<double>(longest * 100 + pressure * 10 + local_density(board, move) * 4);
    score -= static_cast<double>(center_distance_sq(move)) * 0.25;
    return score;
}

// すでに石が置かれているマスのインデックスをすべて集めてリストにして返す
std::vector<int> occupied_indexes(const Board& board) {
    std::vector<int> stones;
    stones.reserve(BOARD_CELLS);
    for (int index = 0; index < BOARD_CELLS; ++index) {
        if (board[static_cast<std::size_t>(index)] != EMPTY) {
            stones.push_back(index);
        }
    }
    return stones;
}

// おかれているすべての石の周囲一マスにある空きマスだけをリストアップして返す
//五目並べでは、遠くに置くのはほぼ無意味であるため。
std::vector<int> neighbor_candidates(const Board& board, int radius = 1) {
    const std::vector<int> stones = occupied_indexes(board);
    if (stones.empty()) {
        return {center_index()};
    }

    std::array<bool, BOARD_CELLS> seen {};
    std::vector<int> candidates;
    for (int index : stones) {
        auto [row, col] = idx_to_rc(index);
        for (int dr = -radius; dr <= radius; ++dr) {
            for (int dc = -radius; dc <= radius; ++dc) {
                const int nr = row + dr;
                const int nc = col + dc;
                if (!inside(nr, nc)) {
                    continue;
                }
                const int candidate = rc_to_idx(nr, nc);
                if (board[static_cast<std::size_t>(candidate)] != EMPTY || seen[static_cast<std::size_t>(candidate)]) {
                    continue;
                }
                seen[static_cast<std::size_t>(candidate)] = true;
                candidates.push_back(candidate);
            }
        }
    }
    return candidates;
}

// 渡された候補手のリストを強い手から順番に並び替える。
void sort_moves_by_heuristic(const Board& board, int player, std::vector<int>& moves) {
    std::stable_sort(
        moves.begin(),
        moves.end(),
        [&](int lhs, int rhs) {
            const double lhs_score = move_shape_score(board, lhs, player);
            const double rhs_score = move_shape_score(board, rhs, player);
            if (lhs_score != rhs_score) {
                return lhs_score > rhs_score;
            }
            const int lhs_center = center_distance_sq(lhs);
            const int rhs_center = center_distance_sq(rhs);
            if (lhs_center != rhs_center) {
                return lhs_center < rhs_center;
            }
            return lhs < rhs;
        }
    );
}

// 現在の盤面における、基本的な合法手のリストを作成して返す
std::vector<int> generate_base_legal_moves(const Board& board, int player) {
    std::vector<int> candidates = neighbor_candidates(board);
    std::vector<int> legal_moves;
    legal_moves.reserve(candidates.size());

    for (int move : candidates) {
        if (board[static_cast<std::size_t>(move)] != EMPTY) {
            continue;
        }
        if (player == BLACK && is_forbidden_for_black(board, move)) {
            continue;
        }
        legal_moves.push_back(move);
    }

    if (legal_moves.empty() && !board_is_full(board)) {
        legal_moves.reserve(BOARD_CELLS);
        for (int move = 0; move < BOARD_CELLS; ++move) {
            if (board[static_cast<std::size_t>(move)] != EMPTY) {
                continue;
            }
            if (player == BLACK && is_forbidden_for_black(board, move)) {
                continue;
            }
            legal_moves.push_back(move);
        }
    }

    sort_moves_by_heuristic(board, player, legal_moves);
    return legal_moves;
}

// 合法手のリストの中から、そこに置けば即ゲーム終了する手があるかチェックし、あればその手のみを返す
std::vector<int> immediate_winning_moves(
    const Board& board,
    int player,
    const std::vector<int>& legal_moves
) {
    std::vector<int> winning_moves;
    for (int move : legal_moves) {
        Board next_board = board_with_move(board, move, player);
        if (winner_after_move(next_board, move, player) == player) {
            winning_moves.push_back(move);
        }
    }
    return winning_moves;
}

// 即勝ち手を最優先し、即負け手をブロックし、候補手の上位candidate_limit個を残して探索対象とする
std::vector<int> generate_policy_moves(const Board& board, int player, int candidate_limit) {
    std::vector<int> legal_moves = generate_base_legal_moves(board, player);
    if (legal_moves.empty()) {
        return {};
    }

    std::vector<int> winning_moves = immediate_winning_moves(board, player, legal_moves);
    if (!winning_moves.empty()) {
        sort_moves_by_heuristic(board, player, winning_moves);
        return winning_moves;
    }

    const int opponent = other_player(player);
    std::vector<int> opponent_legal_moves = generate_base_legal_moves(board, opponent);
    std::vector<int> opponent_wins = immediate_winning_moves(board, opponent, opponent_legal_moves);
    if (!opponent_wins.empty()) {
        std::vector<int> blocking_moves;
        blocking_moves.reserve(opponent_wins.size());
        for (int move : opponent_wins) {
            if (std::find(legal_moves.begin(), legal_moves.end(), move) != legal_moves.end()) {
                blocking_moves.push_back(move);
            }
        }
        if (!blocking_moves.empty()) {
            sort_moves_by_heuristic(board, player, blocking_moves);
            return blocking_moves;
        }
    }

    if (static_cast<int>(legal_moves.size()) > candidate_limit) {
        legal_moves.resize(static_cast<std::size_t>(candidate_limit));
    }
    return legal_moves;
}

// 上位top_k個のなかから、評価値が高い手ほど高確率で選ばれるように重み付きランダムで手を選択する
template <typename URNG>
int choose_weighted_top_move(const Board& board, int player, const std::vector<int>& legal_moves, URNG& rng) {
    if (legal_moves.empty()) {
        throw std::runtime_error("No legal move to choose from.");
    }
    if (legal_moves.size() == 1) {
        return legal_moves.front();
    }

    std::vector<int> ordered = legal_moves;
    sort_moves_by_heuristic(board, player, ordered);

    const int top_k = std::min<int>(4, static_cast<int>(ordered.size()));
    std::vector<double> weights;
    weights.reserve(static_cast<std::size_t>(top_k));
    for (int index = 0; index < top_k; ++index) {
        weights.push_back(static_cast<double>(top_k - index));
    }
    std::discrete_distribution<int> dist(weights.begin(), weights.end());
    return ordered[static_cast<std::size_t>(dist(rng))];
}

// MCTSNodeのメンバ変数群
struct MCTSNode {
    Board board {};
    int player_to_move = BLACK;
    int root_player = BLACK;
    int candidate_limit = DEFAULT_CANDIDATE_LIMIT;
    int move_played = -1;
    MCTSNode* parent = nullptr;
    int terminal_winner = NO_WINNER;
    double wins = 0.0;
    int visits = 0;
    double prior_prob = 1.0; // Prior probability from policy model (default: 1.0)
    std::vector<int> untried_moves;
    std::vector<std::unique_ptr<MCTSNode>> children;

    // コンストラクタ
    MCTSNode(
        const Board& board_value,
        int player_value,
        int root_player_value,
        int candidate_limit_value,
        int move_played_value = -1,
        MCTSNode* parent_value = nullptr,
        int known_winner = NO_WINNER,
        double prior_prob_value = 1.0
    )
        : board(board_value),
          player_to_move(player_value),
          root_player(root_player_value),
          candidate_limit(candidate_limit_value),
          move_played(move_played_value),
          parent(parent_value),
          terminal_winner(known_winner),
          prior_prob(prior_prob_value) {
        if (terminal_winner == NO_WINNER) {
            terminal_winner = board_winner(board);
        }
        if (terminal_winner != NO_WINNER) {
            return;
        }

        untried_moves = generate_policy_moves(board, player_to_move, candidate_limit);
        if (!untried_moves.empty()) {
            return;
        }

        if (board_is_full(board)) {
            terminal_winner = DRAW;
        } else {
            terminal_winner = other_player(player_to_move);
        }
    }

    // この局面が既に勝敗が決した終着点かどうかを返す
    bool is_terminal() const {
        return terminal_winner != NO_WINNER;
    }

    // この局面から打てる候補手がすべて1回以上先読みされたかを判定する
    bool fully_expanded() const {
        return untried_moves.empty();
    }

    // UCT(UCB1)値に基づく最適な子ノードの選択
    MCTSNode* best_child(double exploration) const {
        if (children.empty()) {
            throw std::runtime_error("best_child called on a node with no children.");
        }
        const double sqrt_visits = std::sqrt(static_cast<double>(visits));
        auto score_of = [&](const std::unique_ptr<MCTSNode>& child) {
            const double exploitation = child->wins / static_cast<double>(child->visits);
            const double exploration_term =
                exploration * child->prior_prob * sqrt_visits / (1.0 + child->visits);
            return exploitation + exploration_term;
        };
        const auto best_it = std::max_element(children.begin(), children.end(), [&](const auto& lhs, const auto& rhs) {
            return score_of(lhs) < score_of(rhs);
        });
        return best_it->get();
    }

    // この局面から、まだ探索したことがない手をランダムに1つ選び、それを新しい子ノードとしてツリーに追加して展開する
    template <typename URNG>
    MCTSNode* expand(URNG& rng) {
        if (untried_moves.empty()) {
            throw std::runtime_error("expand called on a fully expanded node.");
        }

        std::uniform_int_distribution<int> index_dist(0, static_cast<int>(untried_moves.size()) - 1);
        const int picked = index_dist(rng);
        const int move = untried_moves[static_cast<std::size_t>(picked)];
        untried_moves.erase(untried_moves.begin() + picked);

        Board next_board = board_with_move(board, move, player_to_move);
        const int winner = winner_after_move(next_board, move, player_to_move);
        children.push_back(std::make_unique<MCTSNode>(
            next_board,
            other_player(player_to_move),
            root_player,
            candidate_limit,
            move,
            this,
            winner
        ));
        return children.back().get();
    }

    // ロールアウトで勝敗が確定した結果を受け取り、このノードの訪問回数と勝利数を更新する
    void update(int winner) {
        ++visits;
        if (winner == DRAW) {
            wins += 0.5;
        } else if (winner == root_player) {
            wins += 1.0;
        }
    }
};

// 探索ツリーの端に達した局面から、ゲームが決着するまで、脳内でモンテカルロ自己対戦を走らせて勝敗結果を予測する
template <typename URNG>
int rollout(Board board, int player, const Options& options, URNG& rng) {
    int current_player = player;

    for (int step = 0; step < options.rollout_limit; ++step) {
        std::vector<int> legal_moves = generate_policy_moves(board, current_player, options.candidate_limit);
        if (legal_moves.empty()) {
            if (board_is_full(board)) {
                return DRAW;
            }
            return other_player(current_player);
        }

        const int move = choose_weighted_top_move(board, current_player, legal_moves, rng);
        board[static_cast<std::size_t>(move)] = current_player;

        const int winner = winner_after_move(board, move, current_player);
        if (winner != NO_WINNER) {
            return winner;
        }
        if (board_is_full(board)) {
            return DRAW;
        }
        current_player = other_player(current_player);
    }

    return DRAW;
}

// 探索の中心的な手順を実行する司令塔関数。simulationsの回数だけ、手の選択から逆伝播まですべて行う
template <typename URNG>
int run_mcts(const Board& board, int player, const Options& options, const std::vector<int>& root_moves, URNG& rng) {
    if (root_moves.empty()) {
        throw std::runtime_error("run_mcts called without legal moves.");
    }
    if (root_moves.size() == 1) {
        return root_moves.front();
    }

    MCTSNode root(board, player, player, options.candidate_limit);
    root.untried_moves = root_moves;

    for (int simulation = 0; simulation < options.simulations; ++simulation) {
        MCTSNode* node = &root;

        while (!node->is_terminal() && node->fully_expanded() && !node->children.empty()) {
            node = node->best_child(options.exploration);
        }

        if (!node->is_terminal() && !node->untried_moves.empty()) {
            node = node->expand(rng);
        }

        int winner = DRAW;
        if (node->is_terminal()) {
            winner = node->terminal_winner;
        } else {
            winner = rollout(node->board, node->player_to_move, options, rng);
        }

        while (node != nullptr) {
            node->update(winner);
            node = node->parent;
        }
    }

    const auto best_it = std::max_element(root.children.begin(), root.children.end(), [](const auto& lhs, const auto& rhs) {
        const double lhs_ratio = lhs->visits > 0 ? lhs->wins / static_cast<double>(lhs->visits) : -1.0;
        const double rhs_ratio = rhs->visits > 0 ? rhs->wins / static_cast<double>(rhs->visits) : -1.0;
        if (lhs->visits != rhs->visits) {
            return lhs->visits < rhs->visits;
        }
        if (lhs_ratio != rhs_ratio) {
            return lhs_ratio < rhs_ratio;
        }
        return center_distance_sq(lhs->move_played) > center_distance_sq(rhs->move_played);
    });

    if (best_it == root.children.end() || (*best_it)->move_played < 0) {
        throw std::runtime_error("MCTS failed to select a move.");
    }
    return (*best_it)->move_played;
}

// 対局中の現在の盤面状態とモデルが選択した指し手をpytorchの学習用データの1行として整形し、バッファ文字列に追加
void append_training_row(std::string& buffer, const Board& board, int move) {
    for (int index = 0; index < BOARD_CELLS; ++index) {
        buffer += std::to_string(board[static_cast<std::size_t>(index)]);
        buffer.push_back(',');
    }
    buffer += std::to_string(SEP_TOKEN_ID);
    buffer.push_back(',');
    buffer += std::to_string(move + MOVE_ID_OFFSET);
    buffer.push_back('\n');
}

// 対局結果の商社を、コンソール表示用にわかりやすいテキストに変換する
std::string winner_label(int winner, bool foul_loss) {
    if (winner == DRAW) {
        return "draw";
    }
    if (winner == BLACK) {
        return "black";
    }
    return foul_loss ? "white(foul)" : "white";
}

// 現在の盤面において、次に打つ手を決定してそのインデックスを返す
template <typename URNG>
int select_move(const Board& board, int player, const Options& options, URNG& rng) {
    const std::vector<int> root_moves = generate_policy_moves(board, player, options.candidate_limit);
    if (root_moves.empty()) {
        throw std::runtime_error("No legal move available.");
    }
    if (root_moves.size() == 1) {
        return root_moves.front();
    }
    if (options.simulations <= 1) {
        return choose_weighted_top_move(board, player, root_moves, rng);
    }
    return run_mcts(board, player, options, root_moves, rng);
}

// 空の盤面からスタートし、決着がつくまで1局を丸ごと実行し、その結果を返す
template <typename URNG>
GameResult play_game(std::uint64_t game_index, int worker_id, const Options& options, URNG& rng, std::ostream& log_stream) {
    Board board {};
    board.fill(EMPTY);

    GameResult result;
    result.csv_rows.reserve(4096);

    int current_player = BLACK;
    while (true) {
        if (options.trace_plies) {
            log_stream << "thread=" << worker_id
                       << " game=" << game_index
                       << " ply=" << (result.plies + 1)
                       << " player=" << (current_player == BLACK ? "black" : "white")
                       << " status=selecting\n";
        }

        const std::vector<int> legal_moves = generate_policy_moves(board, current_player, options.candidate_limit);
        if (legal_moves.empty()) {
            result.winner = board_is_full(board) ? DRAW : other_player(current_player);
            return result;
        }

        const int move = select_move(board, current_player, options, rng);
        append_training_row(result.csv_rows, board, move);
        board[static_cast<std::size_t>(move)] = current_player;
        ++result.plies;

        const int winner = winner_after_move(board, move, current_player);
        if (winner != NO_WINNER) {
            result.winner = winner;
            result.foul_loss = (current_player == BLACK && winner == WHITE);
            return result;
        }

        if (board_is_full(board)) {
            result.winner = DRAW;
            return result;
        }

        current_player = other_player(current_player);
    }
}

// 文字列から数値へ変換する
std::uint64_t parse_u64(const std::string& value, const char* name) {
    try {
        std::size_t processed = 0;
        const unsigned long long parsed = std::stoull(value, &processed, 10);
        if (processed != value.size()) {
            throw std::invalid_argument("trailing characters");
        }
        return static_cast<std::uint64_t>(parsed);
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("Invalid value for ") + name + ": " + value);
    }
}

// 文字列から数値へ変換する
int parse_int(const std::string& value, const char* name) {
    try {
        std::size_t processed = 0;
        const long parsed = std::stol(value, &processed, 10);
        if (processed != value.size()) {
            throw std::invalid_argument("trailing characters");
        }
        if (parsed < std::numeric_limits<int>::min() || parsed > std::numeric_limits<int>::max()) {
            throw std::out_of_range("int overflow");
        }
        return static_cast<int>(parsed);
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("Invalid value for ") + name + ": " + value);
    }
}

// 文字列から数値へ変換する
double parse_double(const std::string& value, const char* name) {
    try {
        std::size_t processed = 0;
        const double parsed = std::stod(value, &processed);
        if (processed != value.size()) {
            throw std::invalid_argument("trailing characters");
        }
        return parsed;
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("Invalid value for ") + name + ": " + value);
    }
}

// python側から渡されるカンマ区切りの盤面状態を分解し、board配列にマッピングする
Board parse_board_str(const std::string& str) {
    Board board {};
    board.fill(EMPTY);
    std::stringstream ss(str);
    std::string item;
    int idx = 0;
    while (std::getline(ss, item, ',')) {
        if (idx >= BOARD_CELLS) {
            throw std::runtime_error("Board string has too many cells.");
        }
        board[static_cast<std::size_t>(idx)] = std::stoi(item);
        idx++;
    }
    if (idx != BOARD_CELLS) {
        throw std::runtime_error("Board string has insufficient cells. Expected 225, got " + std::to_string(idx));
    }
    return board;
}

// プログラムの正しい使い方を表示し、プログラムをその場で終了させる関数
[[noreturn]] void print_usage_and_exit(const char* argv0, int code) {
    std::ostream& stream = code == 0 ? std::cout : std::cerr;
    stream
        << "Usage: " << argv0 << " <games> [--simulations N] [--parallel N] [--seed N]\n"
        << "       [--candidate-limit N] [--rollout-limit N] [--exploration C] [--trace-plies]\n"
        << "       Or in EVAL mode:\n"
        << "       " << argv0 << " --eval --board <csv_str> --move <N> [--simulations N] [--seed N] [--exploration C]\n";
    std::exit(code);
}

// 起動オプションを処理する。--○○みたいなやつを処理する。
Options parse_args(int argc, char* argv[]) {
    if (argc <= 1) {
        print_usage_and_exit(argv[0], 1);
    }

    Options options;
    bool games_set = false;

    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg == "--help" || arg == "-h") {
            print_usage_and_exit(argv[0], 0);
        }
        if (arg == "--trace-plies") {
            options.trace_plies = true;
            continue;
        }
        if (arg == "--eval") {
            options.is_eval = true;
            continue;
        }
        if (arg == "--simulations" || arg == "--parallel" || arg == "--seed" ||
            arg == "--candidate-limit" || arg == "--rollout-limit" || arg == "--exploration" ||
            arg == "--board" || arg == "--move") {
            if (index + 1 >= argc) {
                throw std::runtime_error("Missing value for " + arg);
            }
            const std::string value = argv[++index];
            if (arg == "--simulations") {
                options.simulations = parse_int(value, "--simulations");
            } else if (arg == "--parallel") {
                options.parallel = parse_int(value, "--parallel");
            } else if (arg == "--seed") {
                options.seed = parse_u64(value, "--seed");
                options.has_seed = true;
            } else if (arg == "--candidate-limit") {
                options.candidate_limit = parse_int(value, "--candidate-limit");
            } else if (arg == "--rollout-limit") {
                options.rollout_limit = parse_int(value, "--rollout-limit");
            } else if (arg == "--exploration") {
                options.exploration = parse_double(value, "--exploration");
            } else if (arg == "--board") {
                options.eval_board_str = value;
            } else if (arg == "--move") {
                options.eval_move = parse_int(value, "--move");
            }
            continue;
        }
        if (arg.rfind("--", 0) == 0) {
            throw std::runtime_error("Unknown option: " + arg);
        }
        if (games_set) {
            throw std::runtime_error("Unexpected positional argument: " + arg);
        }
        options.games = parse_u64(arg, "games");
        games_set = true;
    }

    if (!options.is_eval) {
        if (!games_set) {
            throw std::runtime_error("Missing required positional argument: games");
        }
        if (options.games == 0) {
            throw std::runtime_error("games must be positive.");
        }
    } else {
        if (options.eval_board_str.empty()) {
            throw std::runtime_error("EVAL mode requires --board <csv_str>");
        }
        if (options.eval_move < 0 || options.eval_move >= BOARD_CELLS) {
            throw std::runtime_error("EVAL mode requires a valid --move [0-224]");
        }
    }

    if (options.simulations <= 0) {
        throw std::runtime_error("--simulations must be positive.");
    }
    if (options.parallel <= 0) {
        throw std::runtime_error("--parallel must be positive.");
    }
    if (options.candidate_limit <= 0) {
        throw std::runtime_error("--candidate-limit must be positive.");
    }
    if (options.rollout_limit <= 0) {
        throw std::runtime_error("--rollout-limit must be positive.");
    }
    if (!(options.exploration > 0.0)) {
        throw std::runtime_error("--exploration must be greater than 0.");
    }

    return options;
}

// シード値生成
std::uint64_t make_seed(const Options& options, int worker_id) {
    if (options.has_seed) {
        return options.seed + static_cast<std::uint64_t>(worker_id) * 1'000'003ULL;
    }

    std::random_device rd;
    const std::uint64_t base = (static_cast<std::uint64_t>(rd()) << 32) ^ static_cast<std::uint64_t>(rd());
    return base + static_cast<std::uint64_t>(worker_id) * 1'000'003ULL;
}

}  // namespace

// main関数
int main(int argc, char* argv[]) {
    try {
        const Options options = parse_args(argc, argv);

        if (options.is_eval) {
            Board board = parse_board_str(options.eval_board_str);
            int move = options.eval_move;

            if (board[static_cast<std::size_t>(move)] != EMPTY) {
                std::cout << "win_rate=0.0\n";
                return 0;
            }

            const auto [black_count, white_count] = stone_counts(board);
            const int player = (black_count == white_count) ? BLACK : WHITE;

            if (player == BLACK && is_forbidden_for_black(board, move)) {
                std::cout << "win_rate=0.0\n";
                return 0;
            }

            Board next_board = board_with_move(board, move, player);
            const int immediate_winner = winner_after_move(next_board, move, player);

            if (immediate_winner != NO_WINNER) {
                double rate = (immediate_winner == player) ? 1.0 : 0.0;
                std::cout << "win_rate=" << rate << "\n";
                return 0;
            }

            std::mt19937_64 rng(make_seed(options, 0));
            const int next_player = other_player(player);
            const std::vector<int> root_moves = generate_policy_moves(next_board, next_player, options.candidate_limit);

            if (root_moves.empty()) {
                double rate = board_is_full(next_board) ? 0.5 : 0.0;
                std::cout << "win_rate=" << rate << "\n";
                return 0;
            }

            MCTSNode root(next_board, next_player, player, options.candidate_limit);
            root.untried_moves = root_moves;

            for (int simulation = 0; simulation < options.simulations; ++simulation) {
                MCTSNode* node = &root;

                while (!node->is_terminal() && node->fully_expanded() && !node->children.empty()) {
                    node = node->best_child(options.exploration);
                }

                if (!node->is_terminal() && !node->untried_moves.empty()) {
                    node = node->expand(rng);
                }

                int winner = DRAW;
                if (node->is_terminal()) {
                    winner = node->terminal_winner;
                } else {
                    winner = rollout(node->board, node->player_to_move, options, rng);
                }

                while (node != nullptr) {
                    node->update(winner);
                    node = node->parent;
                }
            }

            double win_rate = 0.5;
            if (root.visits > 0) {
                win_rate = root.wins / static_cast<double>(root.visits);
            }
            std::cout << "win_rate=" << win_rate << "\n";
            return 0;
        }

        std::mutex stdout_mutex;
        std::mutex stderr_mutex;
        std::atomic<std::uint64_t> next_game {1};
        std::atomic<std::uint64_t> completed_games {0};

        const int worker_count = options.parallel;
        std::vector<std::thread> workers;
        workers.reserve(static_cast<std::size_t>(worker_count));

        for (int worker_id = 0; worker_id < worker_count; ++worker_id) {
            workers.emplace_back([&, worker_id]() {
                std::mt19937_64 rng(make_seed(options, worker_id));

                {
                    std::lock_guard<std::mutex> lock(stderr_mutex);
                    std::cerr << "thread=" << worker_id << " status=started\n";
                }

                while (true) {
                    const std::uint64_t game_index = next_game.fetch_add(1);
                    if (game_index > options.games) {
                        break;
                    }

                    {
                        std::lock_guard<std::mutex> lock(stderr_mutex);
                        std::cerr << "thread=" << worker_id
                                  << " game=" << game_index
                                  << " status=started\n";
                    }

                    std::ostringstream game_log;
                    GameResult result = play_game(game_index, worker_id, options, rng, game_log);

                    {
                        std::lock_guard<std::mutex> lock(stdout_mutex);
                        std::cout << result.csv_rows;
                    }

                    const std::uint64_t finished = completed_games.fetch_add(1) + 1;
                    {
                        std::lock_guard<std::mutex> lock(stderr_mutex);
                        std::cerr << game_log.str();
                        std::cerr << "thread=" << worker_id
                                  << " game=" << game_index
                                  << " winner=" << winner_label(result.winner, result.foul_loss)
                                  << " plies=" << result.plies
                                  << " completed=" << finished << "/" << options.games
                                  << "\n";
                    }
                }

                {
                    std::lock_guard<std::mutex> lock(stderr_mutex);
                    std::cerr << "thread=" << worker_id << " status=finished\n";
                }
            });
        }

        for (std::thread& worker : workers) {
            worker.join();
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << "\n";
        return 1;
    }
}

// ==========================================
// VCF (Victory by Continuous Fours) Solver
// ==========================================

bool is_winning_move(const Board& board, int move, int player) {
    if (board[static_cast<std::size_t>(move)] != EMPTY) {
        return false;
    }
    if (player == BLACK && is_forbidden_for_black(board, move)) {
        return false;
    }
    Board next_board = board_with_move(board, move, player);
    return winner_after_move(next_board, move, player) == player;
}

bool creates_four_threat(const Board& board, int move, int player) {
    if (board[static_cast<std::size_t>(move)] != EMPTY) {
        return false;
    }
    if (player == BLACK && is_forbidden_for_black(board, move)) {
        return false;
    }

    Board next_board = board_with_move(board, move, player);
    for (int next_move = 0; next_move < BOARD_CELLS; ++next_move) {
        if (is_winning_move(next_board, next_move, player)) {
            return true;
        }
    }
    return false;
}

int solve_vcf_recursive(Board& board, int player, int depth, int& node_count) {
    if (++node_count > 200000) {
        return -1; // 20万ノードで強制打ち切り
    }

    // 1. 即座に勝てる（五連を作れる）手があるかチェック
    std::vector<int> candidates = neighbor_candidates(board, 2);
    for (int move : candidates) {
        if (is_winning_move(board, move, player)) {
            return move;
        }
    }

    if (depth <= 0) {
        return -1;
    }

    const int opponent = other_player(player);

    // 2. 「四」を作る脅威手（スレット）を探索
    std::vector<int> threat_moves;
    for (int move : candidates) {
        if (creates_four_threat(board, move, player)) {
            threat_moves.push_back(move);
        }
    }

    // 各脅威手を深く探索
    for (int move : threat_moves) {
        // 攻撃側の石を置く
        board[static_cast<std::size_t>(move)] = player;

        // 次のターンで攻撃側が即勝利（五連）できる空きマスをカウント
        std::vector<int> win_squares;
        for (int win_move = 0; win_move < BOARD_CELLS; ++win_move) {
            if (is_winning_move(board, win_move, player)) {
                win_squares.push_back(win_move);
            }
        }

        // 勝ちマスが2箇所以上あれば、防御側は防ぎきれないので勝利確定
        if (win_squares.size() >= 2) {
            board[static_cast<std::size_t>(move)] = EMPTY; // undo
            return move;
        }

        // 勝ちマスがちょうど1つの場合、防御側はそこを強制的にブロックしなければならない
        if (win_squares.size() == 1) {
            int block_move = win_squares[0];

            // 防御側にとってブロックする手が合法であるか確認
            if (opponent == BLACK && is_forbidden_for_black(board, block_move)) {
                // 禁手のため防御側は打てない ＝ 攻撃側の勝利確定
                board[static_cast<std::size_t>(move)] = EMPTY; // undo
                return move;
            }

            // 防御側の石を置いてブロックをシミュレート
            board[static_cast<std::size_t>(block_move)] = opponent;

            // ブロック手が相手の勝利にならないかチェック (相手の即勝ちを避けるため)
            if (winner_after_move(board, block_move, opponent) == opponent) {
                // 自滅手なのでこの分岐は無効
                board[static_cast<std::size_t>(block_move)] = EMPTY; // undo block
                board[static_cast<std::size_t>(move)] = EMPTY;       // undo move
                continue;
            }

            // 再帰呼び出し
            int next_win = solve_vcf_recursive(board, player, depth - 1, node_count);

            // 元に戻す
            board[static_cast<std::size_t>(block_move)] = EMPTY; // undo block
            board[static_cast<std::size_t>(move)] = EMPTY;       // undo move

            if (next_win != -1) {
                return move; // 勝ち手順を発見
            }
        } else {
            // 勝ちマスがない（実際にはcreates_four_threatでチェック済みなのでここには来ないはず）
            board[static_cast<std::size_t>(move)] = EMPTY;
        }
    }

    return -1;
}

#ifdef _WIN32
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT
#endif

// モンテカルロ木探索のメインループを実行して、指定した手の評価値を返す関数。
// pythonからこのC APIを呼び出せる
extern "C" {
    // pythonから呼び出せるapi。モンテカルロ木探索のメインループを実行する。
    DLL_EXPORT double run_mcts_c_api(const int* board_array, int move_idx, int simulations, std::uint64_t seed) {
        try {
            Board board;
            for (std::size_t i = 0; i < BOARD_CELLS; ++i) {
                board[i] = board_array[i];
            }
            int move = move_idx;

            if (board[static_cast<std::size_t>(move)] != EMPTY) {
                return 0.0;
            }

            const auto [black_count, white_count] = stone_counts(board);
            const int player = (black_count == white_count) ? BLACK : WHITE;

            if (player == BLACK && is_forbidden_for_black(board, move)) {
                return 0.0;
            }

            Board next_board = board_with_move(board, move, player);
            const int immediate_winner = winner_after_move(next_board, move, player);

            if (immediate_winner != NO_WINNER) {
                double rate = (immediate_winner == player) ? 1.0 : 0.0;
                return rate;
            }

            Options options;
            options.simulations = simulations;
            options.has_seed = true;
            options.seed = seed;

            std::mt19937_64 rng(seed);
            const int next_player = other_player(player);
            const std::vector<int> root_moves = generate_policy_moves(next_board, next_player, options.candidate_limit);

            if (root_moves.empty()) {
                double rate = board_is_full(next_board) ? 0.5 : 0.0;
                return rate;
            }

            MCTSNode root(next_board, next_player, player, options.candidate_limit);
            root.untried_moves = root_moves;

            for (int simulation = 0; simulation < options.simulations; ++simulation) {
                MCTSNode* node = &root;

                while (!node->is_terminal() && node->fully_expanded() && !node->children.empty()) {
                    node = node->best_child(options.exploration);
                }

                if (!node->is_terminal() && !node->untried_moves.empty()) {
                    node = node->expand(rng);
                }

                int winner = DRAW;
                if (node->is_terminal()) {
                    winner = node->terminal_winner;
                } else {
                    winner = rollout(node->board, node->player_to_move, options, rng);
                }

                while (node != nullptr) {
                    node->update(winner);
                    node = node->parent;
                }
            }

            double win_rate = 0.5;
            if (root.visits > 0) {
                win_rate = root.wins / static_cast<double>(root.visits);
            }
            return win_rate;
        } catch (...) {
            return 0.5;
        }
    }

    // 新API: 事前確率(prior_probs)を用いてモデルにガイドされたPUCT探索を行う
    DLL_EXPORT double run_mcts_c_api_with_policy(
        const int* board_array, 
        int move_idx, 
        int simulations, 
        std::uint64_t seed,
        const double* prior_probs
    ) {
        try {
            Board board;
            for (std::size_t i = 0; i < BOARD_CELLS; ++i) {
                board[i] = board_array[i];
            }
            int move = move_idx;

            if (board[static_cast<std::size_t>(move)] != EMPTY) {
                return 0.0;
            }

            const auto [black_count, white_count] = stone_counts(board);
            const int player = (black_count == white_count) ? BLACK : WHITE;

            if (player == BLACK && is_forbidden_for_black(board, move)) {
                return 0.0;
            }

            Board next_board = board_with_move(board, move, player);
            const int immediate_winner = winner_after_move(next_board, move, player);

            if (immediate_winner != NO_WINNER) {
                double rate = (immediate_winner == player) ? 1.0 : 0.0;
                return rate;
            }

            Options options;
            options.simulations = simulations;
            options.has_seed = true;
            options.seed = seed;

            std::mt19937_64 rng(seed);
            const int next_player = other_player(player);
            const std::vector<int> root_moves = generate_policy_moves(next_board, next_player, options.candidate_limit);

            if (root_moves.empty()) {
                double rate = board_is_full(next_board) ? 0.5 : 0.0;
                return rate;
            }

            MCTSNode root(next_board, next_player, player, options.candidate_limit);
            root.untried_moves = root_moves;

            for (int simulation = 0; simulation < options.simulations; ++simulation) {
                MCTSNode* node = &root;

                while (!node->is_terminal() && node->fully_expanded() && !node->children.empty()) {
                    node = node->best_child(options.exploration);
                }

                if (!node->is_terminal() && !node->untried_moves.empty()) {
                    MCTSNode* parent_node = node;
                    node = node->expand(rng);
                    // ルートの直下に展開したノードには、渡された事前確率を割り当てる
                    if (parent_node == &root && prior_probs != nullptr) {
                        node->prior_prob = prior_probs[node->move_played];
                    }
                }

                int winner = DRAW;
                if (node->is_terminal()) {
                    winner = node->terminal_winner;
                } else {
                    winner = rollout(node->board, node->player_to_move, options, rng);
                }

                while (node != nullptr) {
                    node->update(winner);
                    node = node->parent;
                }
            }

            double win_rate = 0.5;
            if (root.visits > 0) {
                win_rate = root.wins / static_cast<double>(root.visits);
            }
            return win_rate;
        } catch (...) {
            return 0.5;
        }
    }

    // 新API: 現在の局面から単一のMCTS探索を実行し、各手の訪問回数を visits_out に書き戻す
    DLL_EXPORT double run_mcts_c_api_with_policy_and_visits(
        const int* board_array, 
        int simulations, 
        std::uint64_t seed,
        const double* prior_probs,
        int* visits_out
    ) {
        try {
            Board board;
            for (std::size_t i = 0; i < BOARD_CELLS; ++i) {
                board[i] = board_array[i];
            }

            // 訪問回数配列の初期化
            if (visits_out != nullptr) {
                std::fill(visits_out, visits_out + BOARD_CELLS, 0);
            }

            const auto [black_count, white_count] = stone_counts(board);
            const int player = (black_count == white_count) ? BLACK : WHITE;

            Options options;
            options.simulations = simulations;
            options.has_seed = true;
            options.seed = seed;

            std::mt19937_64 rng(seed);
            const std::vector<int> root_moves = generate_policy_moves(board, player, options.candidate_limit);

            if (root_moves.empty()) {
                return 0.5;
            }

            // 現在の局面(board)をルートとする
            MCTSNode root(board, player, other_player(player), options.candidate_limit);
            root.untried_moves = root_moves;

            for (int simulation = 0; simulation < options.simulations; ++simulation) {
                MCTSNode* node = &root;

                while (!node->is_terminal() && node->fully_expanded() && !node->children.empty()) {
                    node = node->best_child(options.exploration);
                }

                if (!node->is_terminal() && !node->untried_moves.empty()) {
                    MCTSNode* parent_node = node;
                    node = node->expand(rng);
                    // ルートの直下に展開したノードには、渡された事前確率を割り当てる
                    if (parent_node == &root && prior_probs != nullptr) {
                        node->prior_prob = prior_probs[node->move_played];
                    }
                }

                int winner = DRAW;
                if (node->is_terminal()) {
                    winner = node->terminal_winner;
                } else {
                    winner = rollout(node->board, node->player_to_move, options, rng);
                }

                while (node != nullptr) {
                    node->update(winner);
                    node = node->parent;
                }
            }

            // ルートの子ノードの訪問回数を visits_out に書き戻す
            if (visits_out != nullptr) {
                for (const auto& child : root.children) {
                    if (child->move_played >= 0 && child->move_played < BOARD_CELLS) {
                        visits_out[child->move_played] = child->visits;
                    }
                }
            }

            double win_rate = 0.5;
            if (root.visits > 0) {
                win_rate = root.wins / static_cast<double>(root.visits);
            }
            return win_rate;
        } catch (...) {
            return 0.5;
        }
    }

    // VCF (Victory by Continuous Fours) 探索用の C-API
    DLL_EXPORT int solve_vcf_c_api(const int* board_array, int player, int max_depth) {
        try {
            Board board;
            for (std::size_t i = 0; i < BOARD_CELLS; ++i) {
                board[i] = board_array[i];
            }
            int node_count = 0;
            return solve_vcf_recursive(board, player, max_depth, node_count);
        } catch (...) {
            return -1;
        }
    }
}

