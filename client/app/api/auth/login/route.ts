import prisma from "@/lib/prisma";
import bcrypt from "bcryptjs";
import { NextResponse } from "next/server";

interface RequestBody {
  username: string;
  password: string;
}

export default async function POST(request: Request) {
  const body: RequestBody = await request.json();

  const user = await prisma.user.findUnique({
    where: { username: body.username },
  });

  if (user && (await bcrypt.compare(body.password, user.password))) {
    const { password, ...userWithoutPass } = user;
    return NextResponse.json(userWithoutPass);
  } else return NextResponse.json(null);
}
