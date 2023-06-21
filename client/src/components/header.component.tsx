"use client";

import { signOut, useSession } from "next-auth/react";
import Link from "next/link";

const Header = () => {
  const { data: session } = useSession();
  const user = session?.user;

  return (
    <header className="h-20 w-screen">
      <nav className="navbar bg-base-100 px-8">
        <div className="flex-1">
          <Link
            href="/"
            className="link link-primary link-hover hover:no-underline text-4xl font-semibold"
          >
            VectorVerse
          </Link>
        </div>
        <div className="flex-none">
          <ul className="menu menu-horizontal px-2">
            <li>
              <Link href="/" className="text-xl">
                Home
              </Link>
            </li>
            {!user && (
              <>
                <li>
                  <Link href="/login" className="text-xl">
                    Login
                  </Link>
                </li>
                <li>
                  <Link href="/register" className="text-xl">
                    Register
                  </Link>
                </li>
              </>
            )}
            {user && (
              <>
                <li>
                  <Link href="/profile" className="text-xl">
                    Profile
                  </Link>
                </li>
                <li onClick={() => signOut()}>
                  <a className="text-xl">Logout</a>
                </li>
              </>
            )}
            <li>
              <Link href="/vector" className="text-xl">Vector</Link>
            </li>
          </ul>
        </div>
      </nav>
    </header>
  );
};

export default Header;
