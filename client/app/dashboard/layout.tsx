"use client";
import NavBar from "../../components/NavBar";

export default function dashboard({ children }: { children: React.ReactNode }) {
  return (
    <>
      <NavBar />
      {children}
    </>
  );
}
