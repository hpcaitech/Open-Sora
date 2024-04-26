import { GrDocumentPdf, GrGithub } from "react-icons/gr";

const Footer = () => {
  return (
    <footer className="py-8 bg-slate-300 text-center">
      <div className="text-center">
        <div className="flex justify-center space-x-4">
          <a
            href="https://nerfies.github.io/static/videos/nerfies_paper.pdf"
            target="_blank"
            rel="noopener noreferrer"
          >
            <GrDocumentPdf size={24} />
          </a>
          <a
            href="https://github.com/keunhong"
            target="_blank"
            rel="noopener noreferrer"
          >
            <GrGithub size={24} />
          </a>
        </div>
        <div>
          <div className="flex justify-center mx-auto text-center">
            <div className="p-4 md:w-2/3 lg:w-1/2 text-left">
              <p>
                This website is licensed under a{" "}
                <a
                  rel="license"
                  href="http://creativecommons.org/licenses/by-sa/4.0/"
                >
                  Creative Commons Attribution-ShareAlike 4.0 International
                  License
                </a>
                .
              </p>
              <p>
                This means you are free to borrow the{" "}
                <a href="https://github.com/nerfies/nerfies.github.io">
                  source code
                </a>{" "}
                of this website, we just ask that you link back to this page in
                the footer. Please remember to remove the analytics code
                included in the header of the website which you do not want on
                your website.
              </p>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
