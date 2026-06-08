import sqlite3
import os

def fix_db():
    db_path = "mlflow.db"
    if not os.path.exists(db_path):
        print(f"Error: {db_path} not found. Please run this script from the project root directory.")
        return

    # 現在の絶対パスを取得し、スラッシュ区切りにする
    current_dir = os.getcwd().replace('\\', '/')
    print(f"Current workspace path: {current_dir}")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    old_prefix_file = "file:///content/renjuTransformer_grpo"
    old_prefix_raw = "/content/renjuTransformer_grpo"
    new_prefix = f"file:///{current_dir}"

    print(f"\nReplacing '{old_prefix_file}' with '{new_prefix}'...")
    print(f"Replacing '{old_prefix_raw}' with '{new_prefix}'...")

    # experiments テーブルの更新
    cur.execute(
        "UPDATE experiments SET artifact_location = replace(artifact_location, ?, ?)",
        (old_prefix_file, new_prefix)
    )
    cur.execute(
        "UPDATE experiments SET artifact_location = replace(artifact_location, ?, ?)",
        (old_prefix_raw, new_prefix)
    )

    # runs テーブルの更新
    cur.execute(
        "UPDATE runs SET artifact_uri = replace(artifact_uri, ?, ?)",
        (old_prefix_file, new_prefix)
    )
    cur.execute(
        "UPDATE runs SET artifact_uri = replace(artifact_uri, ?, ?)",
        (old_prefix_raw, new_prefix)
    )

    # コミットと結果表示
    conn.commit()
    
    # 確認表示
    print("\nUpdated experiments:")
    for row in cur.execute("SELECT experiment_id, artifact_location FROM experiments"):
        print(row)
        
    print("\nUpdated runs (first 5):")
    for row in cur.execute("SELECT run_uuid, experiment_id, artifact_uri FROM runs LIMIT 5"):
        print(row)

    conn.close()
    print("\nDatabase update completed successfully!")

if __name__ == "__main__":
    fix_db()
