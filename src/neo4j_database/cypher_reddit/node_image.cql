UNWIND $reddits as reddit
WITH reddit
WHERE reddit.post_hint = 'image'
MERGE (n:Image {url:reddit.url})
