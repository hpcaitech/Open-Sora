const VideoItem = ({ video }) => {
  return (
    <div>
      <div
        style={{
          position: "relative",
          width: "100%",
          height: "0px",
          "padding-bottom": "100.000%",
        }}
      >
        <iframe
          title={video.prompt}
          allow="fullscreen;autoplay"
          allowfullscreen
          height="100%"
          src={video.link}
          width="100%"
          style={{
            border: "none",
            width: "100%",
            height: "100%",
            position: "absolute",
            left: "0px",
            top: "0px",
            overflow: "hidden",
          }}
        ></iframe>
      </div>
      <p className="leading-tight text-justify text-sm">{video.prompt}</p>
    </div>
  );
};

const VideoGrid = ({ videoGroup }) => {
  return (
    <div>
      <span className="text-3xl text-semibold">{videoGroup.title}</span>
      <div className="grid mt-4 gap-4 grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4">
        {videoGroup.items.map((video) => (
          <VideoItem video={video} />
        ))}
      </div>
    </div>
  );
};

export default VideoGrid;
