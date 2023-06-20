"use client";
import { signIn as NextAuthSignIn } from "next-auth/react";
import Link from "next/link";
import { FormEvent, useRef } from "react";

export default async function SignInPage() {
  const username = useRef<string>();
  const password = useRef<string>();

  const signInWithCredentials = (event: FormEvent<HTMLFormElement>) => {
    NextAuthSignIn("credentials", {
      username: username.current,
      password: password.current,
      redirect: false,
      callbackUrl: "/",
    });
  };

  return (
    <div className="flex items-center justify-center h-full w-screen">
      <div className="flex flex-col rounded-md outline shadow-lg outline-1 p-4">
        <form className="flex flex-col" onSubmit={signInWithCredentials}>
          <h1 className="mb-4">Login</h1>
          <div className="mb-3 inline-flex items-center">
            <label htmlFor="username" className="w-full">
              Username:
            </label>
            <input
              id="username"
              className="ml-2 input input-primary input-bordered"
              type="text"
              onChange={(e) => (username.current = e.target.value)}
            />
          </div>
          <div className="mb-4 inline-flex items-center">
            <label className="w-full" htmlFor="password">
              Password:
            </label>
            <input
              id="password"
              className="ml-2 input input-primary input-bordered"
              type="password"
              onChange={(e) => (password.current = e.target.value)}
            />
          </div>
          <button className="btn btn-primary mb-1" type="submit">
            Log in
          </button>
          <div className="flex justify-end">
            <Link href={"/auth/user"}>Don't have an account?</Link>
          </div>
          <div className="text-center my-4">
            <div className="inline-block align-middle w-[40%] border-t-2 border-gray-400"></div>
            <span className="align-middle px-2">or</span>
            <div className="inline-block align-middle w-[40%] border-t-2 border-gray-400"></div>
          </div>
        </form>
        <button
          className="btn btn-secondary mb-2"
          onClick={() => NextAuthSignIn("google")}
        >
          Sign in with Google
        </button>
        <button
          className="btn btn-secondary"
          onClick={() => NextAuthSignIn("github")}
        >
          Sign in with Github
        </button>
      </div>
    </div>
  );
}
