import requests
import sys
import time

API_TOKEN = "your_api_token_here"
API_URL = "https://your.awx-instance.com/api/"

def start_workflow(playbook_id, inventory_id=None, job_type="run"):
    """Start an AWX workflow job"""
    url = f"{API_URL}jobs/"
    
    payload = {
        "extra_vars": {},
        "launch_source": "api",
        "launch_options": {}
    }
    
    if inventory_id:
        payload.update({
            "inventory_id": inventory_id,
            "job_type": job_type
        })
    
    response = requests.post(url, json=payload, auth=(API_TOKEN,))
    
    if response.status_code == 201:
        return response.json()
    else:
        print(f"Failed to start workflow: {response.text}")
        return None

def approve_workflow(job_id):
    """Approve a workflow job"""
    url = f"{API_URL}jobs/{job_id}/"
    
    response = requests.patch(url, json={"approve": True}, auth=(API_TOKEN,))
    
    return response.status_code == 200

def main():
    # Prompt user for inputs
    playbook_id = input("Enter playbook ID: ")
    inventory_id = input("Enter inventory ID (or press Enter for none): ") or None
    job_type = input("Enter job type (run, import, etc.): ") or "run"
    approve = input("Type 'yes' to proceed: ")
    
    if approve.lower() != "yes":
        print("Operation cancelled by user")
        sys.exit(0)
    
    print("Starting workflow job...")
    job = start_workflow(playbook_id, inventory_id, job_type)
    
    if job:
        job_id = job["id"]
        print(f"Job started with ID: {job_id}")
        
        print("Approving job...")
        if approve_workflow(job_id):
            print("Job approved successfully!")
        else:
            print("Failed to approve job")
    else:
        print("Failed to start workflow")

if __name__ == "__main__":
    main()
