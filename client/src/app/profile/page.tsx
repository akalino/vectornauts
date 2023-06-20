import { getServerSession } from "next-auth";
import Header from "@/components/header.component";
import { authOptions } from "@/lib/auth";

export default async function Profile() {
  const session = await getServerSession(authOptions);
  const user = session?.user;

  return (
    <>
      <Header />
      <section className="min-h-screen flex justify-center w-screen items-center">
        <div className="p-4 card h-fit flex justify-center items-center">
          <div>
            <h1 className="mb-0">Profile</h1>
            {!user ? (
              <p>Loading...</p>
            ) : (
              <div className="flex items-center gap-8">
                <div>
                  <img
                    src={user.image ? user.image : "/images/default.png"}
                    className="max-h-36"
                    alt={`profile photo of ${user.name}`}
                  />
                </div>
                <div>
                  <p className="text-xl mb-3">
                    <b className="mr-2">Name:</b>
                    {user.name}
                  </p>
                  <p className="text-xl mb-3">
                    <b className="mr-2">Email:</b>
                    {user.email}
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      </section>
    </>
  );
}
