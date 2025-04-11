from huggingface_hub import HfApi, create_repo
import os
import sys

def upload_to_huggingface():
    # Get environment variables
    space_id = os.environ.get('HF_SPACE_NAME')
    token = os.environ.get('HF_TOKEN')
    
    if not space_id or not token:
        print("Error: Missing required environment variables")
        sys.exit(1)

    try:
        # Create/update the space with correct SDK
        create_repo(
            space_id,
            repo_type='space',
            space_sdk='gradio',
            token=token,
            exist_ok=True
        )

        # Upload the content
        api = HfApi()
        api.upload_folder(
            folder_path='space_content',
            repo_id=space_id,
            repo_type='space',
            token=token
        )
        print(f"Successfully updated space: {space_id}")
    except Exception as e:
        print(f"Error updating space: {e}")
        sys.exit(1)

if __name__ == "__main__":
    upload_to_huggingface()
