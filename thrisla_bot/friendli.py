import requests

#non-chat completions endpoint
url = "https://api.friendli.ai/dedicated/v1/completions"


payload = {
    "model": "bw0uym5j6696",
    "prompt": "Hello how are you?",
    "max_tokens": 100
}

headers = {
    "Authorization": "Bearer flp_2c1BLYbIBYp2Q6sbZcVSbGZFs81DzVsoRPT2C5b6CQjMb8",
    "Content-Type": "application/json"
}

#post request to api
response = requests.post(url, json=payload, headers=headers)

print(response.text)
