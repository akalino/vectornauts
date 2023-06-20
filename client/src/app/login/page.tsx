import { LoginForm } from "./form";
import Header from "@/components/header.component";

export default function LoginPage() {
  return (
    <>
      <Header />
      <section className="min-h-screen pt-20 w-screen">
        <div className="mx-auto px-6 py-12 h-full flex justify-center items-center">
          <div className="card md:w-8/12 lg:w-5/12 px-8 py-10">
            <LoginForm />
          </div>
        </div>
      </section>
    </>
  );
}
