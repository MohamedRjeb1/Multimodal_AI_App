"""
Migration script to move old files to the new structure.
"""
import os
import shutil
from pathlib import Path


def migrate_files():
    """Migrate old files to new structure."""
    print("Starting migration of old files...")
    
    # Create backup directory
    backup_dir = Path("backup_old_structure")
    backup_dir.mkdir(exist_ok=True)
    
    # Files to migrate
    migrations = [
        # Move old model files to backup
        ("model/youtube.py", "backup_old_structure/youtube.py"),
        ("model/transcript.py", "backup_old_structure/transcript.py"),
        ("model/llm_generate.py", "backup_old_structure/llm_generate.py"),
        ("model/textLoader.py", "backup_old_structure/textLoader.py"),
        ("model/test_audio.py", "backup_old_structure/test_audio.py"),
        
        # Move old routes to backup
        ("routes/base.py", "backup_old_structure/base.py"),
        ("routes/url.py", "backup_old_structure/url.py"),
        
        # Move old helpers to backup
        ("helpers/config.py", "backup_old_structure/config.py"),
        
        # Move old main.py to backup
        ("main.py", "backup_old_structure/main.py"),
    ]
    
    # Perform migrations
    for old_path, new_path in migrations:
        if os.path.exists(old_path):
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            
            # Copy file to backup
            shutil.copy2(old_path, new_path)
            print(f"Backed up: {old_path} -> {new_path}")
        else:
            print(f"File not found: {old_path}")
    
    # Move data files to new structure
    data_migrations = [
        ("files/audio", "data/audio"),
        ("files/transcripts", "data/transcripts"),
    ]
    
    for old_path, new_path in data_migrations:
        if os.path.exists(old_path):
            # Create new directory
            os.makedirs(new_path, exist_ok=True)
            
            # Move files
            for file in os.listdir(old_path):
                old_file = os.path.join(old_path, file)
                new_file = os.path.join(new_path, file)
                shutil.move(old_file, new_file)
                print(f"Moved: {old_file} -> {new_file}")
            
            # Remove old directory if empty
            try:
                os.rmdir(old_path)
                print(f"Removed empty directory: {old_path}")
            except OSError:
                print(f"Directory not empty, keeping: {old_path}")
        else:
            print(f"Directory not found: {old_path}")
    
    print("\nMigration completed!")
    print("Old files have been backed up to 'backup_old_structure/'")
    print("Data files have been moved to the new 'data/' structure")
    print("\nYou can now safely delete the old directories:")
    print("- model/")
    print("- routes/")
    print("- helpers/")
    print("- files/")


if __name__ == "__main__":
    migrate_files()
