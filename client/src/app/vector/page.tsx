import Header from "@/components/header.component";
import {PineconeClient} from "@pinecone-database/pinecone";

const pinecone = new PineconeClient();

export default async function Index() {
    // TODO: Wrap around login
    // TODO: Also wrap with swr
    await pinecone.init({
        environment: process.env.PINECONE_ENV as string,
        apiKey: process.env.PINECONE_API_KEY as string,
    });
    const indices = await pinecone.listIndexes();
    return (
        <>
            <Header/>
            <section className="min-h-screen pt-20 w-screen">
                <div className="container mx-auto px-4">
                    <div className="grid gap-4 md:grid-cols-4">
                        {indices.map((vector, i) => (
                            <div className="card w-50 bg-base-100 shadow-xl" key={i}>
                                <div className="card-body">
                                    <h2 className="card-title">{vector}</h2>
                                    <div className="card-actions justify-end">
                                        <button className="btn btn-primary">View</button>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </section>
        </>
    );
}
