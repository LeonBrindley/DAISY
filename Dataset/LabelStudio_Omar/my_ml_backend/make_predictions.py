import requests
import json

# Define the endpoint and headers
url = "http://51.20.55.222:8080/api/dm/actions?id=retrieve_tasks_predictions&tabID=2&project=2"
headers = {
    "Authorization": "Token c620059cb3e58794123bcb6eceb0363cf475e070",
    "Content-Type": "application/json"
}

# Function to create the payload with the given start ID
def create_payload(start_id):
    ids = list(range(start_id, start_id + 500))
    payload = {
        "ordering": ["tasks:id"],
        "selectedItems": {
            "all": False,
            "included": ids
        },
        "project": "2"
    }
    return payload

# Function to send a batch of predictions
def send_batch_predictions(start_id):
    payload = create_payload(start_id)
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    return response.status_code, response.json()

# Main loop to perform predictions
# start_id = 18156
start_id = 45158
end_id = 114653
# end_id = 19156

while start_id <= end_id:
    status_code, response_json = send_batch_predictions(start_id)
    print(f"Sent batch starting with ID {start_id}: Status Code {status_code}")
    # Optionally, print the response or handle it as needed
    print(response_json)
    
    start_id += 500

