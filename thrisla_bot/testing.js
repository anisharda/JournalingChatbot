const options = {
    method: 'POST',
    headers: {Authorization: 'Bearer flp_2c1BLYbIBYp2Q6sbZcVSbGZFs81DzVsoRPT2C5b6CQjMb8', 'Content-Type': 'application/json'},
    body: '{"messages":[{"content":"You are a helpful assistant.","role":"system"},{"content":"Hello!","role":"user"}],"model":"bw0uym5j6696"}'
  };
  
  fetch('https://api.friendli.ai/dedicated/v1/chat/completions', options)
    .then(response => response.json())
    .then(response => console.log(response))
    .catch(err => console.error(err));