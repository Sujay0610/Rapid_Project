import requests
import os
import json

bearer_token = 'AAAAAAAAAAAAAAAAAAAAAJC3ugEAAAAAjpcv3ek8bsySdTTNQE3q7wbnfTg%3DXRRsPVGmI1AXqiWPgtvk9aJAox1MUQBk0kfIm8KhK5VS8XkbPY'

search_url = "https://api.twitter.com/2/tweets/search/all"
query_params = {
    'query': 'bitcoin -is:retweet lang:en',
    'start_time': '2021-01-01T00:00:00Z',
    'end_time': '2024-12-31T23:59:59Z',
    'max_results': 500,
    'tweet.fields': 'id,text,created_at,author_id'
}

def create_headers(bearer_token):
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    return headers

def connect_to_endpoint(url, headers, params):
    response = requests.request("GET", url, headers=headers, params=params)
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()

headers = create_headers(bearer_token)
json_response = connect_to_endpoint(search_url, headers, query_params)

# Save tweets to a file
with open('bitcoin_tweets_academic.json', 'w') as file:
    json.dump(json_response, file)
