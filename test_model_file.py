# check_registry.py
import os
import mlflow #type: ignore
from mlflow.tracking import MlflowClient #type: ignore

# 1. Connect to your local database path
db_path = os.path.abspath("mlflow.db")
mlflow.set_tracking_uri(f"sqlite:///{db_path}")

client = MlflowClient()

print("--- LOCAL MLFLOW REGISTRY AUDIT ---")
try:
    # List all registered model names in your database
    registered_models = client.search_registered_models()
    if not registered_models:
        print("[⚠️ WARNING] No registered models found in mlflow.db. Your registry is completely empty!")

    for rm in registered_models:
        print(f"\nModel Name Found: '{rm.name}'")

        # Check all available versions
        versions = client.search_model_versions(f"name='{rm.name}'")
        for v in versions:
            # Check what aliases are attached to this version
            aliases = client.get_model_version_by_alias(rm.name, "Production") if "Production" in v.aliases else None
            print(f"  -> Version {v.version} (Run ID: {v.run_id}) | Aliases: {v.aliases}")

except Exception as e:
    print(f"[💥 ERROR] Failed to read database: {str(e)}")
