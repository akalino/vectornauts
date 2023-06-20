"use client";
import { signOut } from "next-auth/react";
import Link from "next/link";

import React from "react";

const NavBar = () => {
  return (
    <div>
      <Link
        key="signOut"
        onClick={() => signOut()}
        href={String(process.env.LOCAL_AUTH_URL)}
      >
        TEST
      </Link>
    </div>
  );
};

export default NavBar;
