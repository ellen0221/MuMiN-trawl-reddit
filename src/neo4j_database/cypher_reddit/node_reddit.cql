UNWIND $reddits AS reddit
WITH reddit
WHERE reddit.id IS NOT NULL
MERGE (n:Reddit {redditId:reddit.id})
ON CREATE SET
    n.text = reddit.selftext,
    n.title = reddit.title,
    n.createdAt = reddit.created_utc,
    n.ups = toInteger(reddit.ups),
    n.commentCount = toInteger(reddit.num_comments),
    n.crosspostsCount = toInteger(reddit.num_crossposts);
