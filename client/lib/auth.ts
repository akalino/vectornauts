import NextAuth, { AuthOptions } from "next-auth";
import GoogleProvider from "next-auth/providers/google";
import GithubProvider from "next-auth/providers/github";
import Credentials from "next-auth/providers/credentials";
import axios from "axios";

export const authOptions: AuthOptions = {
  // Configure one or more authentication providers
  providers: [
    Credentials({
      name: "Credentials",
      credentials: {
        username: { label: "Username", type: "text" },
        password: { label: "Password", type: "password" },
      },
      authorize: async (credentials) => {
        const user = await axios
          .post("/api/login", {
            username: credentials?.username,
            password: credentials?.password,
          })
          .then((res) => res.data);

        if (user) {
          return { id: String(user.id), email: user.email };
        }

        return null;
      },
    }),
    GoogleProvider({
      clientId: String(process.env.GOOGLE_ID),
      clientSecret: String(process.env.GOOGLE_SECRET),
    }),
    GithubProvider({
      clientId: String(process.env.GITHUB_ID),
      clientSecret: String(process.env.GITHUB_SECRET),
    }),
  ],
  pages: {
    signIn: "/auth/signin",
  },
  // callbacks: {
  //   async signIn({ user }) {
  //     let isAllowedToSignIn = true;
  //     console.log(user);
  //     return isAllowedToSignIn;
  //   },
  // },
};

export default NextAuth(authOptions);
