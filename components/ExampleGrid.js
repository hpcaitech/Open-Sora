import { GrAdd, GrLinkNext } from "react-icons/gr";
import classNames from "classnames";

const VideoItem = ({ video }) => {
  return (
    <div className="w-64">
      <div
        style={{
          position: "relative",
          width: "100%",
          height: "0px",
          paddingBottom: "100.000%",
        }}
      >
        <iframe
          title={video.link}
          allow="fullscreen;autoplay"
          allowFullScreen
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
    </div>
  );
};

const renderItem = (item) => {
  const hasInputs = item.inputs.length > 0;

  return (
    <div className={hasInputs ? "w-full" : "w-64"}>
      <div className="flex flex-wrap justify-center items-center space-x-4">
        {hasInputs &&
          item.inputs.map((input_, idx) => {
            if (idx < item.inputs.length - 1) {
              return (
                <>
                  <VideoItem video={input_} />
                  <GrAdd size={64} />
                </>
              );
            } else {
              return (
                <>
                  <VideoItem video={input_} />
                  <GrLinkNext size={64} />
                </>
              );
            }
          })}

        <VideoItem video={item.output} />
      </div>

      <div className="flex justify-center">
        <p className="leading-tight text-justify text-sm">{item.prompt}</p>
      </div>
    </div>
  );
};

const ExampleGrid = ({ data }) => {
  return (
    <div className="mb-4">
      <span className="text-3xl text-semibold">{data.title}</span>
      <div className="flex flex-wrap justify-around">
        {data.items.map((item) => renderItem(item))}
      </div>
    </div>
  );
};

export default ExampleGrid;
