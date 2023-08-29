UNWIND $reddits AS reddit
WITH reddit
WHERE reddit.subreddit_id IS NOT NULL AND reddit.id IS NOT NULL
MATCH (r:Reddit {redditId:reddit.id})
MATCH (s:Subreddit {subredditId:reddit.subreddit_id})
MERGE (r)-[:BELONGS_TO]->(s);
