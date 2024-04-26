import "./App.css";
import Title from "./components/Title";
import PubButtonList from "./components/PubButtonList";
import pub_links from "./data/pub_links";
import examples from "./data/examples";
import ExampleGrid from "./components/ExampleGrid";
import Footer from "./components/Footer";

function App() {
  return (
    <div className="text-center">
      {/* head section */}
      <div>
        <div className="py-4">
          <Title />
        </div>

        <div className="text-3xl font-semibold py-6">
          <span>
            Open-Sora: Democratizing Efficient Video Production for All
          </span>
        </div>

        <div>
          <PubButtonList links={pub_links} />
        </div>
      </div>

      {/* video sections */}
      <div className="container mx-auto my-8 px-8 md:px-16 md:pt-12 lg:px-36">
        {examples.map((exampleGroup) => (
          <ExampleGrid data={exampleGroup} />
        ))}
      </div>

      {/* Footer */}
      <div className="mt-8">
        <Footer />
      </div>
    </div>
  );
}

export default App;
