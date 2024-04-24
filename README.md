# Open-Sora Gallery

We rewrite the `nerfies`[https://github.com/google/nerfies] with React to enable people without knowledge in html to edit the web content by simply changing the data files.

The web content will display videos given in the `src/data/video.js` file. This file contains a `videos` variable and its structure follows this format:

```javascript
[
    // video group
    {
        "title": "group1"
        "items": [
            // put videos here
            {
                "prompt": "some descriptions",
                "link": "link to the video 1"
            },
            {
                "prompt": "some descriptions",
                "link": "link to the video 2"
            },
        ]
    },

    // another group of videos
    {
        "title": "group2"
        "items": [
            // put videos here
            {
                "prompt": "some descriptions",
                "link": "link to the video 3"
            },
            {
                "prompt": "some descriptions",
                "link": "link to the video 4"
            },
        ]
    },
]

```

If you wish to add another video, just add the following data to the `items` field in a video group.

```javascript
{
    "prompt": "your prompt",
    "link": "video url"
}
```

If you wish to add another video group, just replicate the group structure and edit the video items inside.