import { Html, Head, Main, NextScript } from "next/document";

export default function Document() {
  return (
    <Html>
      <Head>
        <meta charSet="utf-8" />
        <meta
          name="description"
          content="Gallery for Open-Sora, a project for democratizing efficient video production for all."
        />
        <meta
          name="keywords"
          content="Open-Sora, Video Generation, AIGC, GenerativeAI, Open-Source, Video Production"
        ></meta>
        <meta name="theme-color" content="#000000" />
        <meta property="og:type" content="website" />
        <meta property="og:title" content="Open-Sora Gallery" />
        <meta
          property="og:description"
          content="Gallery for Open-Sora, a project for democratizing efficient video production for all."
        />
        <meta property="og:site_name" content="Open-Sora Gallery" />
        <link rel="icon" href="/favicon.ico" />
        <link rel="manifest" href="/manifest.json" />
      </Head>
      <body>
        <Main />
        <NextScript />
      </body>
    </Html>
  );
}
