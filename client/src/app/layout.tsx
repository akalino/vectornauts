import "./globals.css";
import { NextAuthProvider } from "./providers";

export const metadata = {
  title: "Create Next App",
  description: "Generated by create next app",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="prose">
        <NextAuthProvider>{children}</NextAuthProvider>
      </body>
    </html>
  );
}
