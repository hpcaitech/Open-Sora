# Open-Sora Gallery

We rewrite the `nerfies`[https://github.com/google/nerfies] with React and Next.js to enable people without knowledge in html to edit the web content by simply changing the data files.

## 🚀 Deploy

## ✏️ Edit Examples

The web content will display demo examples given in the `data/examples.js` file. This file contains an `examples` variable and its structure follows this format:

```javascript
[
  // video examples
  {
    title: "Text To Video",
    items: [
      {
        prompt: "some prompt",
        inputs: [],
        output: {
          link: "link to a video on Streamable",
        },
      },
    ],
  },

  // another group of examples
  {
    title: "Animating Image",
    items: [
      {
        prompt: "some prompt",
        inputs: [
          {
            link: "link to a video on Streamable",
          },
        ],
        output: {
          link: "link to a video on Streamable",
        },
      },
    ],
  },
];
```

If you wish to add another video, you can basically append an object like the one below to the `items` field. This defines a single example and the fields are explained below.

```javascript
{
    prompt: "some prompt",
    inputs: [],
    output: {
        link: "link to a video on Streamable",
    },
}
```

- prompt: the prompt used to generate the video
- inputs: it is list of objects in the format of `{ link: "link to a video on Streamable" }`, these are the reference images/videos used to generate the final video. Streamable can display images as well.
- output: it is a object in the format of `{ link: "link to a video on Streamable" }`, this is the final video

Some examples for difference generation cases are given below:

1. Text to Video

```javascript
{
    prompt: "some prompt",
    inputs: [

    ],
    output: {
        link: "link to a video on Streamable",
    },
}
```

2. Image to Video

```javascript
{
    prompt: "some prompt",
    inputs: [
        {
            link: "link to an image on Streamable",
        }
    ],
    output: {
        link: "link to a video on Streamable",
    },
}
```

3. Image Connecting

```javascript
{
    prompt: "some prompt",
    inputs: [
        {
            link: "link to an image on Streamable",
        },
        {
            link: "link to an image on Streamable",
        }
    ],
    output: {
        link: "link to a video on Streamable",
    },
}
```

4. Video Connecting

```javascript
{
    prompt: "some prompt",
    inputs: [
        {
            link: "link to a video on Streamable",
        },
        {
            link: "link to a video on Streamable",
        }
    ],
    output: {
        link: "link to a video on Streamable",
    },
}
```
