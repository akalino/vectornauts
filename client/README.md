# VectorVerse

## Getting Started
First install NodeJS 18
```shell
cd ~
curl -s https://deb.nodesource.com/setup_18.x | sudo bash
sudo apt install nodejs -y
node -v
```

Set these environment variables:
```console
DATABASE_URL="file:../../data/db/vectorverse.sqlite"
```

To install the necessary packages and build the prisma db client, from the **client** directort run:
```shell
npm install
npx prisma generate
prisma migrate dev --name init
npx prisma db push
```



Do not login with Github or Google, this requires secrets set in your environment variables which you will not have access to.

## Run the App
To run the app:
```shell
npm run dev
```