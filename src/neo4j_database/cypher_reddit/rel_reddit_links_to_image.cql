UNWIND $reddits AS reddit
WITH reddit
WHERE reddit.url IS NOT NULL AND reddit.id IS NOT NULL AND reddit.post_hint = 'image'
MATCH (r:Reddit {redditId:reddit.id})
MATCH (i:Image {url:reddit.url})
MERGE (r)-[:HAS_IMAGE]->(i);
