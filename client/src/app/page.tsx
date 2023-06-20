import Header from "@/components/header.component";

export default async function Home() {
  return (
    <>
      <Header />
      <section className="min-h-screen pt-20 w-screen">
        <div className="max-w-4xl mx-auto rounded-md h-[20rem] flex justify-center items-center">
          <p className="text-3xl font-semibold">Welcome to VectorVerse</p>
        </div>
      </section>
    </>
  );
}
