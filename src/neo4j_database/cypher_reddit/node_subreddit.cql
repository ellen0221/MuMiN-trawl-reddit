UNWIND $reddits AS reddit
WITH reddit
WHERE reddit.subreddit_id IS NOT NULL
MERGE (n:Subreddit {subredditId:reddit.subreddit_id})
ON CREATE SET
    n.name = reddit.subreddit_name_prefixed;
