UNWIND $reddits AS reddit
WITH reddit
WHERE reddit.url IS NOT NULL AND reddit.id IS NOT NULL AND reddit.post_hint = 'link'
MATCH (r:Reddit {redditId:reddit.id})
MATCH (u:Url {name:reddit.url})
MERGE (r)-[:LINKS_TO]->(u);
