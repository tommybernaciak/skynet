import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
import pandas as pd
from datetime import datetime

openai.api_key = "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"

starttime = datetime.now()
# products dataframe
products = [
  {
    "Product ID": "1001",
    "Product Name": "CHILI PUFFER HOOD - Vinterjakker",
    "Description": "warm winter jacket",
    "Brand": "Jack & Jones",
    "Main Image URL": "https://img01.ztat.net/article/spp-media-p1/6b4981cdff524fb39061b687a609601e/3c959d1ed3944a75816f5528186edace.jpg?imwidth=1800",
    "Additional Image URLs": "",
    "Base Price": "519,00",
    "Sale Price": "",
    "Currency": "DKK",
    "Primary Category": "Men",
    "Sub-category": "Jacket",
    "Tags": "winter,warm",
    "Size": "M/L/XL",
    "Color": "Black/Blue/Green",
    "Material": "Polyester",
    "Average Rating": 4.5,
    "Reviews": "So warm!",
    "Vendor Name": "Zalando",
    "Vendor ID": "V001",
    "Active Promotions": "",
    "Direct URL to marketplace": "https://www.zalando.dk/jack-and-jones-jjchili-puffer-hood-vinterjakker-schwarz-ja222t10f-q12.html"
  },
  {
    "Product ID": "1002",
    "Product Name": "AIGLE WINTER JACKETS AIW23MOUI028 - Vinterjakker",
    "Description": "warm winter jacket",
    "Brand": "Aigle",
    "Main Image URL": "https://img01.ztat.net/article/spp-media-p1/437fdca62b494bf08b65289640e0bcdc/c3bd2f99fafb41e3a52e630fed749136.jpg?imwidth=1800&filter=packshot",
    "Additional Image URLs": "",
    "Base Price": "2,835,90",
    "Sale Price": "",
    "Currency": "DKK",
    "Primary Category": "Men",
    "Sub-category": "Jacket",
    "Tags": "winter,warm",
    "Size": "M/L",
    "Color": "Khaki",
    "Material": "Polyamid",
    "Average Rating": 4.2,
    "Reviews": "Perfect for winter!",
    "Vendor Name": "Zalando",
    "Vendor ID": "V001",
    "Active Promotions": "",
    "Direct URL to marketplace": "https://www.zalando.dk/aigle-vinterjakker-black-ai222t03q-q11.html"
  },
  {
    "Product ID": "1003",
    "Product Name": "PIKE LAKE™ HOODED JACKET - WINTER JACKET - Vinterjakker",
    "Description": "casual winter jacket",
    "Brand": "Columbia",
    "Main Image URL": "https://img01.ztat.net/article/spp-media-p1/8ca563495ca23772a535919f052e106f/0031e7fc4868413ab0f609ec15147342.jpg?imwidth=1800",
    "Additional Image URLs": "",
    "Base Price": "1,495,00",
    "Sale Price": "1,271,00",
    "Currency": "DKK",
    "Primary Category": "Men",
    "Sub-category": "Jacket",
    "Tags": "winter,warm, casual",
    "Size": "S/M/L/XL",
    "Color": "Black/Shark/Olive Green",
    "Material": "Polyester",
    "Average Rating": 4.3,
    "Reviews": "Kept me warm",
    "Vendor Name": "Zalando",
    "Vendor ID": "V001",
    "Active Promotions": "",
    "Direct URL to marketplace": "https://www.zalando.dk/columbia-outdoor-jakke-black-c2342f01y-q11.html"
  },
  {
    "Product ID": "1004",
    "Product Name": "adidas Handball Spezial",
    "Description": "sport sneakers, regular fit, lace closure, leather upper, textile lining",
    "Brand": "adidas",
    "Main Image URL": "https://images.stockx.com/images/adidas-Handball-Spezial-Cloud-White.jpg?fit=fill&bg=FFFFFF&w=576&h=384&fm=avif&auto=compress&dpr=1&trim=color&updated_at=1694240903&q=57",
    "Additional Image URLs": "",
    "Base Price": "115,00",
    "Sale Price": "",
    "Currency": "USD",
    "Primary Category": "Unisex",
    "Sub-category": "Sneakers",
    "Tags": "sport, casual",
    "Size": "36,37,38,39,40,41,42",
    "Color": "Cloud White",
    "Material": "",
    "Average Rating": 4.1,
    "Reviews": "So stylish!",
    "Vendor Name": "StockX",
    "Vendor ID": "V002",
    "Active Promotions": "",
    "Direct URL to marketplace": "https://stockx.com/en-gb/adidas-handball-spezial-cloud-white"
  },
  {
    "Product ID": "1005",
    "Product Name": "adidas 3 Vulc Shoes",
    "Description": "sport sneakers, regular fit, lace closure, Vulcanised rubber outsole.",
    "Brand": "adidas",
    "Main Image URL": "https://media.handball-store.com/catalog/product/cache/image/1800x/9df78eab33525d08d6e5fb8d27136e95/a/d/adidas-originals_b22705_1_footwear_photography_side_lateral_center_view_white_000.jpg",
    "Additional Image URLs": "",
    "Base Price": "66,22",
    "Sale Price": "56,28",
    "Currency": "USD",
    "Primary Category": "Unisex",
    "Sub-category": "Sneakers",
    "Tags": "sport, casual",
    "Size": "36,37,38,39,40,41,42",
    "Color": "White",
    "Material": "",
    "Average Rating": 4.3,
    "Reviews": "Very stylish!",
    "Vendor Name": "Handball-Store",
    "Vendor ID": "V003",
    "Active Promotions": "",
    "Direct URL to marketplace": "https://handball-store.com/b22705-adidas-3-vulc-shoes-white-white-metallic-gold"
  },
  {
    "Product ID": "1006",
    "Product Name": "adidas HANDBALL SPEZIAL SHOES",
    "Description": "sport sneakers, regular fit, lace closure, leather upper, textile lining",
    "Brand": "adidas",
    "Main Image URL": "https://assets.adidas.com/images/h_840,f_auto,q_auto,fl_lossy,c_fill,g_auto/27bf1e7ef90d4624a547d86d54dac34e_9366/Handball_Spezial_Shoes_White_IE9837_01_standard.jpg",
    "Additional Image URLs": "",
    "Base Price": "110,00",
    "Sale Price": "",
    "Currency": "USD",
    "Primary Category": "Unisex",
    "Sub-category": "Sneakers",
    "Tags": "sport, casual",
    "Size": "36,37,38,39,40,41,42",
    "Color": "Cloud White",
    "Material": "",
    "Average Rating": 4.2,
    "Reviews": "Fits well!",
    "Vendor Name": "Adidas",
    "Vendor ID": "V004",
    "Active Promotions": "",
    "Direct URL to marketplace": "https://www.adidas.com/us/handball-spezial-shoes/IE9837.html"
  },
  {
    "Product ID": "1007",
    "Product Name": "Columbia Women's Heavenly Hooded Jacket",
    "Description": "casual winter jacket",
    "Brand": "Columbia",
    "Main Image URL": "https://m.media-amazon.com/images/I/616N1h1bLZS._AC_UX569_.jpg",
    "Additional Image URLs": "",
    "Base Price": "150,00",
    "Sale Price": "140,00",
    "Currency": "USD",
    "Primary Category": "Women",
    "Sub-category": "Jacket",
    "Tags": "winter,warm, casual",
    "Size": "S/M/L/XL",
    "Color": "Black/Red/Pink",
    "Material": "Polyester",
    "Average Rating": 4.7,
    "Reviews": "Amazingly warm and lightweight",
    "Vendor Name": "Amazon",
    "Vendor ID": "V005",
    "Active Promotions": "",
    "Direct URL to marketplace": "https://www.amazon.com/Columbia-Womens-Heavenly-Hooded-Jacket/dp/B01MYU9QR7/ref=sr_1_3?qid=1697711618&refinements=p_89%3AColumbia&s=apparel&sr=1-3&th=1&psc=1"
  },
  {
    "Product ID": "1008",
    "Product Name": "Columbia Men's Ascender Softshell Front-zip Jacket",
    "Description": "casual winter jacket, water resistant",
    "Brand": "Columbia",
    "Main Image URL": "https://m.media-amazon.com/images/I/71gluQqwAYL._AC_UY550_.jpg",
    "Additional Image URLs": "",
    "Base Price": "115,00",
    "Sale Price": "",
    "Currency": "USD",
    "Primary Category": "Men",
    "Sub-category": "Jacket",
    "Tags": "winter,warm, casual",
    "Size": "S/M/L/XL",
    "Color": "Black/Red/Delta/Graphite",
    "Material": "Polyester",
    "Average Rating": 4.7,
    "Reviews": "Great jacket for everyday use at a great price!",
    "Vendor Name": "Amazon",
    "Vendor ID": "V005",
    "Active Promotions": "",
    "Direct URL to marketplace": "https://www.amazon.com/Columbia-Ascender-Softshell-Jacket-Resistant/dp/B00HQ4KS3W/ref=sr_1_6?keywords=Columbia%2BBlack%2BJacket&qid=1697711646&sr=8-6&th=1&psc=1"
  },
  {
    "Product ID": "1009",
    "Product Name": "Columbia Men's Delta Ridge Down Jacket",
    "Description": "casual winter jacket",
    "Brand": "Columbia",
    "Main Image URL": "https://m.media-amazon.com/images/I/71M7mEwpmLL._AC_UX569_.jpg",
    "Additional Image URLs": "",
    "Base Price": "150,00",
    "Sale Price": "138,79",
    "Currency": "USD",
    "Primary Category": "Men",
    "Sub-category": "Jacket",
    "Tags": "warm, casual",
    "Size": "S/M/L/XL",
    "Color": "Black/Navy/Green",
    "Material": "Polyester",
    "Average Rating": 4.7,
    "Reviews": "Perfect!",
    "Vendor Name": "Amazon",
    "Vendor ID": "V005",
    "Active Promotions": "",
    "Direct URL to marketplace": "https://www.amazon.com/Columbia-Delta-Ridge-Jacket-Medium/dp/B07JBZ627V/ref=sr_1_26?keywords=Columbia+Black+Jacket&qid=1697711646&sr=8-26"
  },
  {
    "Product ID": "1010",
    "Product Name": "CLUB TEE - T-shirts basic",
    "Description": "regular cotton tshirt",
    "Brand": "Nike",
    "Main Image URL": "https://img01.ztat.net/article/spp-media-p1/e24b03d53a714675a6ced6dd68125818/757070488a5b46b8838e282a1654f019.jpg?imwidth=1800",
    "Additional Image URLs": "",
    "Base Price": "199,00",
    "Sale Price": "",
    "Currency": "DKK",
    "Primary Category": "Men",
    "Sub-category": "T-shirt",
    "Tags": "casual, sport, everyday, regular",
    "Size": "S/M/L/XL",
    "Color": "White/Pink/Aligator/Baltic",
    "Material": "Cotton",
    "Average Rating": 4.3,
    "Reviews": "Cool!",
    "Vendor Name": "Zalando",
    "Vendor ID": "V001",
    "Active Promotions": "",
    "Direct URL to marketplace": "https://www.zalando.dk/nike-sportswear-club-tee-t-shirts-basic-ni122o0ce-a12.html"
  },
  {
    "Product ID": "1011",
    "Product Name": "SIMPLE DOME TEE - Sports T-shirts",
    "Description": "regular cotton tshirt",
    "Brand": "The North Face",
    "Main Image URL": "https://img01.ztat.net/article/spp-media-p1/5a2054521ea63c76b967129ac00f99e0/53e05d695047435790cf3ffaf5b8e17c.jpg?imwidth=1800",
    "Additional Image URLs": "",
    "Base Price": "249,00",
    "Sale Price": "224,00",
    "Currency": "DKK",
    "Primary Category": "Men",
    "Sub-category": "T-shirt",
    "Tags": "casual, everyday, regular",
    "Size": "S/M/L",
    "Color": "White/Black/Pink/Orange/Gravel",
    "Material": "Cotton",
    "Average Rating": 4.5,
    "Reviews": "So Stylish!",
    "Vendor Name": "Zalando",
    "Vendor ID": "V001",
    "Active Promotions": "",
    "Direct URL to marketplace": "https://www.zalando.dk/the-north-face-simple-dome-t-shirts-basic-white-th342d00t-a11.html"
  },
  {
    "Product ID": "1012",
    "Product Name": "ORIGINAL - T-shirts basic",
    "Description": "regular cotton tshirt",
    "Brand": "GANT",
    "Main Image URL": "https://img01.ztat.net/article/spp-media-p1/1aae619ddfed3881adf4291320b3489c/008ba9180eb842c89e12d309be4b4d6f.jpg?imwidth=1800",
    "Additional Image URLs": "",
    "Base Price": "349,00",
    "Sale Price": "",
    "Currency": "DKK",
    "Primary Category": "Men",
    "Sub-category": "T-shirt",
    "Tags": "casual, everyday, regular",
    "Size": "S/M/L/XL",
    "Color": "White/Azul/Beige/Black",
    "Material": "Cotton",
    "Average Rating": 4.0,
    "Reviews": "It's ok",
    "Vendor Name": "Zalando",
    "Vendor ID": "V001",
    "Active Promotions": "",
    "Direct URL to marketplace": "https://www.zalando.dk/gant-t-shirts-basic-hvid-ga322d00x-a11.html"
  },
  {
    "Product ID": "1013",
    "Product Name": "Comfort Colors Men's Short Sleeve Tee, Style 1717",
    "Description": "Adult tee knitted of the softest 100% cotton",
    "Brand": "Comfort Colors",
    "Main Image URL": "https://m.media-amazon.com/images/I/51f1nf53sDL._AC_UX466_.jpg",
    "Additional Image URLs": "",
    "Base Price": "9,79",
    "Sale Price": "",
    "Currency": "USD",
    "Primary Category": "Men",
    "Sub-category": "T-shirt",
    "Tags": "casual, everyday, regular",
    "Size": "S/M/L/XL",
    "Color": "White/Black/Navy",
    "Material": "Cotton",
    "Average Rating": 4.6,
    "Reviews": "This is an amazing basic shirt",
    "Vendor Name": "Amazon",
    "Vendor ID": "V005",
    "Active Promotions": "",
    "Direct URL to marketplace": "https://www.amazon.com/Comfort-Colors-Sleeve-1717-X-Large/dp/B07M8NX6YB/ref=sr_1_7?crid=1K8EPYVRNPNJU&keywords=white%2Btshirt&qid=1698048662&sprefix=white%2Btshir%2Caps%2C224&sr=8-7&th=1&psc=1"
  },
  {
    "Product ID": "1014",
    "Product Name": "Hanes Men's Undershirts, Odor Control, Moisture-Wicking Tee Shirts, Multi-Packs",
    "Description": "slim-fit tee that has all the best features of a classic cotton T-shirt but with the pared-down silhouette you prefer.",
    "Brand": "Hanes",
    "Main Image URL": "https://m.media-amazon.com/images/I/71+fiqvTvEL._AC_UX569_.jpg",
    "Additional Image URLs": "",
    "Base Price": "10,98",
    "Sale Price": "",
    "Currency": "USD",
    "Primary Category": "Men",
    "Sub-category": "T-shirt",
    "Tags": "casual, everyday, regular",
    "Size": "S/M/L/XL",
    "Color": "White",
    "Material": "Cotton",
    "Average Rating": 4.5,
    "Reviews": "High-Quality Comfort at an Affordable Price",
    "Vendor Name": "Amazon",
    "Vendor ID": "V005",
    "Active Promotions": "",
    "Direct URL to marketplace": "https://www.amazon.com/Hanes-Available-Moisture-Wicking-Undershirts-Multipack/dp/B000HOBYCW/ref=sr_1_10?crid=1K8EPYVRNPNJU&keywords=white%2Btshirt&qid=1698048662&sprefix=white%2Btshir%2Caps%2C224&sr=8-10&th=1&psc=1"
  },
  {
        "Product ID": "1015",
        "Product Name": "TEE FUTURA UNISEX - T-shirts basic",
        "Description": "regular cotton tshirt",
        "Brand": "Nike",
        "Main Image URL": "https://img01.ztat.net/article/spp-media-p1/b37c72e1184c3acb8cab529d8b7a644c/2490d6d4805443b4b4481dee0e96ac0c.jpg?imwidth=1800",
        "Additional Image URLs": "",
        "Base Price": "149,00",
        "Sale Price": "",
        "Currency": "DKK",
        "Primary Category": "Men",
        "Sub-category": "T-shirt",
        "Tags": "casual, sport, everyday, regular",
        "Size": "S/M/L/XL",
        "Color": "Black/Khaki/Blue",
        "Material": "Cotton",
        "Average Rating": "4.6",
        "Reviews": "Fits well!",
        "Vendor Name": "Zalando",
        "Vendor ID": "V001",
        "Active Promotions": "",
        "Direct URL to marketplace": "https://www.zalando.dk/nike-sportswear-tee-futura-t-shirts-basic-ni126g00b-q11.html"
    },
    {
        "Product ID": "1016",
        "Product Name": "TEEN SIMPLE DOME TEE UNISEX - T-shirts basic",
        "Description": "regular cotton tshirt",
        "Brand": "The North Face",
        "Main Image URL": "https://img01.ztat.net/article/spp-media-p1/826cfa2afb0148669c4c8b7703580fdf/6081a2072f5f453d8874552003ee6479.jpg?imwidth=1800&filter=packshot",
        "Additional Image URLs": "",
        "Base Price": "219,00",
        "Sale Price": "197,00",
        "Currency": "DKK",
        "Primary Category": "Unisex",
        "Sub-category": "T-shirt",
        "Tags": "casual, everyday, regular",
        "Size": "S/M/L/XL",
        "Color": "Black/Orange/White",
        "Material": "Cotton",
        "Average Rating": "4.4",
        "Reviews": "Amazing!",
        "Vendor Name": "Zalando",
        "Vendor ID": "V001",
        "Active Promotions": "",
        "Direct URL to marketplace": "https://www.zalando.dk/the-north-face-teen-simple-dome-tee-t-shirts-basic-black-th343d01r-q11.html"
    },
    {
        "Product ID": "1017",
        "Product Name": "SPORTS T-SHIRT - Sports T-shirts",
        "Description": "sport cotton t-shirt",
        "Brand": "Lacoste",
        "Main Image URL": "https://img01.ztat.net/article/spp-media-p1/e1c6ae4ffa0e3f1baae2937f441d63eb/3af4963ba6e14151974450a893e492ef.jpg?imwidth=1800&filter=packshot",
        "Additional Image URLs": "",
        "Base Price": "199,00",
        "Sale Price": "179,00",
        "Currency": "DKK",
        "Primary Category": "Men",
        "Sub-category": "T-shirt",
        "Tags": "casual, sport",
        "Size": "S/M/L/XL",
        "Color": "Black/Blue/Green",
        "Material": "Cotton/Polyester",
        "Average Rating": "4.0",
        "Reviews": "So stylish!",
        "Vendor Name": "Zalando",
        "Vendor ID": "V001",
        "Active Promotions": "",
        "Direct URL to marketplace": "https://www.zalando.dk/lacoste-sport-tennis-funktionstrojer-l0643d006-q11.html"
    },
    {
        "Product ID": "1018",
        "Product Name": "SUPERSTAR - Sneakers",
        "Description": "sport sneakers, regular fit",
        "Brand": "adidas",
        "Main Image URL": "https://img01.ztat.net/article/spp-media-p1/28e3d724a9103476bbf8180c33486e59/6ecf856da30e4872bd4dde0a2c3a61bd.jpg?imwidth=1800&filter=packshot",
        "Additional Image URLs": "",
        "Base Price": "559,00",
        "Sale Price": "",
        "Currency": "DKK",
        "Primary Category": "Men",
        "Sub-category": "Sneakers",
        "Tags": "sport, casual",
        "Size": "36",
        "Color": "White",
        "Material": "",
        "Average Rating": "4.2",
        "Reviews": "Perfect!",
        "Vendor Name": "Zalando",
        "Vendor ID": "V001",
        "Active Promotions": "",
        "Direct URL to marketplace": "https://www.zalando.dk/adidas-originals-superstar-sneakers-hvid-ad116d00a-a11.html"
    },
    {
        "Product ID": "1019",
        "Product Name": "SUPERSTAR UNISEX - Sneakers",
        "Description": "sport sneakers, regular fit",
        "Brand": "adidas",
        "Main Image URL": "https://img01.ztat.net/article/spp-media-p1/9ef2f62a55773ee2ac0bf313b8718b66/5d77c27f12264dc3a3bb8b54ccb4d3a7.jpg?imwidth=1800",
        "Additional Image URLs": "",
        "Base Price": "579,00",
        "Sale Price": "",
        "Currency": "DKK",
        "Primary Category": "Unisex",
        "Sub-category": "Sneakers",
        "Tags": "sport, casual",
        "Size": "35,36,37",
        "Color": "White/Black",
        "Material": "",
        "Average Rating": "4.0",
        "Reviews": "It's ok",
        "Vendor Name": "Zalando",
        "Vendor ID": "V001",
        "Active Promotions": "",
        "Direct URL to marketplace": "https://www.zalando.dk/adidas-originals-superstar-sneakers-footwear-white-ad116d0qb-a11.html"
    },
    {
        "Product ID": "1020",
        "Product Name": "Gopune Men's Windproof Bomber Jacket Winter Warm Padded Outwear Flight Coats",
        "Description": "casual winter jacket",
        "Brand": "Gopune",
        "Main Image URL": "https://m.media-amazon.com/images/I/71NzKGqEo5L._AC_SX569_.jpg",
        "Additional Image URLs": "",
        "Base Price": "45,99",
        "Sale Price": "",
        "Currency": "USD",
        "Primary Category": "Men",
        "Sub-category": "Jacket",
        "Tags": "casual",
        "Size": "S/M/L/XL",
        "Color": "Black",
        "Material": "Polyester",
        "Average Rating": "4.3",
        "Reviews": "Good for the money",
        "Vendor Name": "Amazon",
        "Vendor ID": "V005",
        "Active Promotions": "",
        "Direct URL to marketplace": "https://www.amazon.com/Gopune-Bomber-Jacket-Military-Outwear/dp/B0BCG9N331/ref=sr_1_35?crid=3910ZAU5BMH2&keywords=winter+jackets+for+men&qid=1699008146&sprefix=winter+jacket%2Caps%2C191&sr=8-35"
    },
    {
        "Product ID": "1021",
        "Product Name": "TACVASEN Men's Skiing Jacket with Hood Waterproof Hiking Fishing Travel Fleece Jacket Parka Raincoat",
        "Description": "casual winter jacket",
        "Brand": "Tacvasen",
        "Main Image URL": "https://m.media-amazon.com/images/I/61H7TD6sXjL._AC_SX466_.jpg",
        "Additional Image URLs": "",
        "Base Price": "87,99",
        "Sale Price": "64,98",
        "Currency": "USD",
        "Primary Category": "Men",
        "Sub-category": "Jacket",
        "Tags": "casual",
        "Size": "S/M/L/XL",
        "Color": "Black/Blue/Green",
        "Material": "Polyester",
        "Average Rating": "4.6",
        "Reviews": "Warm, perfect",
        "Vendor Name": "Amazon",
        "Vendor ID": "V005",
        "Active Promotions": "",
        "Direct URL to marketplace": "https://www.amazon.com/TACVASEN-Jacket-Windproof-Fleece-Winter/dp/B07Z7CSGF4/ref=sr_1_24?crid=3910ZAU5BMH2&keywords=winter+jackets+for+men&qid=1699008146&sprefix=winter+jacket%2Caps%2C191&sr=8-24"
    },
    {
        "Product ID": "1022",
        "Product Name": "TACVASEN Men's Jacket-Casual Winter Cotton Military Jacket Thicken Hooded Cargo Coat",
        "Description": "casual winter jacket",
        "Brand": "Tacvasen",
        "Main Image URL": "https://m.media-amazon.com/images/I/610+Si1DRuL._AC_SX522_.jpg",
        "Additional Image URLs": "",
        "Base Price": "108,99",
        "Sale Price": "75,98",
        "Currency": "USD",
        "Primary Category": "Men",
        "Sub-category": "Jacket",
        "Tags": "casual",
        "Size": "S/M/L/XL",
        "Color": "Black",
        "Material": "Cotton/Polyester",
        "Average Rating": "4.4",
        "Reviews": "Excellent Coat!",
        "Vendor Name": "Amazon",
        "Vendor ID": "V005",
        "Active Promotions": "",
        "Direct URL to marketplace": "https://www.amazon.com/TACVASEN-Jackets-Winter-Military-Outdoor/dp/B07L57PG34/ref=sr_1_15?crid=3910ZAU5BMH2&keywords=winter+jackets+for+men&qid=1699008146&sprefix=winter+jacket%2Caps%2C191&sr=8-15"
    },
    {
        "Product ID": "1023",
        "Product Name": "Hanes Originals Lightweight, Crewneck T-Shirts for Men, Tri-Blend Tee",
        "Description": "regular cotton tshirt",
        "Brand": "Hanes",
        "Main Image URL": "https://m.media-amazon.com/images/I/81rODzF1tOL._AC_SX569_.jpg",
        "Additional Image URLs": "",
        "Base Price": "14,00",
        "Sale Price": "9,00",
        "Currency": "USD",
        "Primary Category": "Men",
        "Sub-category": "T-shirt",
        "Tags": "casual, everyday, regular",
        "Size": "S/M/L/XL",
        "Color": "Black/White/Navy",
        "Material": "Polyester/Cotton",
        "Average Rating": "4.0",
        "Reviews": "Fits Nicely!",
        "Vendor Name": "Amazon",
        "Vendor ID": "V005",
        "Active Promotions": "",
        "Direct URL to marketplace": "https://www.amazon.com/Hanes-Originals-Lightweight-Crewneck-Tri-Blend/dp/B0BTTQNRTH/ref=sr_1_18?crid=33TFPVN12IBFW&keywords=black+tshirt+for+men&qid=1699008067&sprefix=black+tshirt+for+men%2Caps%2C197&sr=8-18"
    },
    {
        "Product ID": "1024",
        "Product Name": "Russell Athletic Men's Dri-Power Cotton Blend Tees & Tanks",
        "Description": "regular cotton tshirt",
        "Brand": "Russell Athletics",
        "Main Image URL": "https://m.media-amazon.com/images/I/61KtrldtmeL._AC_SY550_.jpg",
        "Additional Image URLs": "",
        "Base Price": "9,99",
        "Sale Price": "6,81",
        "Currency": "USD",
        "Primary Category": "Men",
        "Sub-category": "T-shirt",
        "Tags": "casual, everyday, regular",
        "Size": "XL",
        "Color": "Black",
        "Material": "Cotton/Polyester",
        "Average Rating": "4.2",
        "Reviews": "Nice Fit",
        "Vendor Name": "Amazon",
        "Vendor ID": "V005",
        "Active Promotions": "",
        "Direct URL to marketplace": "https://www.amazon.com/Russell-Athletic-Essential-Short-Sleeve/dp/B071DVQKVK/ref=sr_1_15?crid=33TFPVN12IBFW&keywords=black%2Btshirt%2Bfor%2Bmen&qid=1699008067&sprefix=black%2Btshirt%2Bfor%2Bmen%2Caps%2C197&sr=8-15&th=1&psc=1"
    },
    {
    "Product ID": "1025",
    "Product Name": "Hanes Men's Originals T-Shirt, 100% Cotton Tees for Men",
    "Description": "regular cotton tshirt",
    "Brand": "Hanes",
    "Main Image URL": "https://m.media-amazon.com/images/I/81pR-PtZX7L._AC_SX569_.jpg",
    "Additional Image URLs": "",
    "Base Price": "13,00",
    "Sale Price": "8,79",
    "Currency": "USD",
    "Primary Category": "Men",
    "Sub-category": "T-shirt",
    "Tags": "casual, everyday, regular",
    "Size": "S/M/L/XL",
    "Color": "Black/White",
    "Material": "Cotton",
    "Average Rating": "4.2",
    "Reviews": "Very comfortable",
    "Vendor Name": "Amazon",
    "Vendor ID": "V005",
    "Active Promotions": "",
    "Direct URL to marketplace": "https://www.amazon.com/Hanes-Originals-Sleeve-Crewneck-T-Shirt/dp/B0BY3J5L6F/ref=sr_1_6?crid=33TFPVN12IBFW&keywords=black%2Btshirt%2Bfor%2Bmen&qid=1699008067&sprefix=black%2Btshirt%2Bfor%2Bmen%2Caps%2C197&sr=8-6&th=1&psc=1"
},
{
    "Product ID": "1026",
    "Product Name": "Amazon Essentials Men's Crewneck Undershirt, Pack of 6",
    "Description": "regular cotton tshirt",
    "Brand": "Amazon",
    "Main Image URL": "https://m.media-amazon.com/images/I/51v5brDQozS._AC_SX466_.jpg",
    "Additional Image URLs": "",
    "Base Price": "26,00",
    "Sale Price": "",
    "Currency": "USD",
    "Primary Category": "Men",
    "Sub-category": "T-shirt",
    "Tags": "casual, everyday, regular",
    "Size": "S/M/L/XL",
    "Color": "White",
    "Material": "Cotton",
    "Average Rating": "4.4",
    "Reviews": "Good value for money",
    "Vendor Name": "Amazon",
    "Vendor ID": "V005",
    "Active Promotions": "",
    "Direct URL to marketplace": "https://www.amazon.com/Amazon-Essentials-6-Pack-Crewneck-Undershirts/dp/B06XW85TSB/ref=sr_1_1_ffob_sspa?crid=1KVHDY3JER8QA&keywords=white%2Btshirt%2Bfor%2Bmen&qid=1699008058&sprefix=white%2Btshirt%2Bfor%2Bmen%2Caps%2C243&sr=8-1-spons&sp_csd=d2lkZ2V0TmFtZT1zcF9hdGY&th=1&psc=1"
}
]
product_data_df = pd.DataFrame(products)

# user = {"user_id": "U001", "user_name": "John Doe", "gender": "man", "age": "30", "height": "180", "weight": "80", "shoe_size": "42", "clothing_size": "L", "preferred_color": "black", "preferred_material": "polyester", "preferred_brands": ["Columbia"], "excluded_vendors": ["Zalando"]}
# print("User data:")
# print(user)


# add combined column and text embedding column
product_data_df['combined'] = product_data_df.apply(lambda row: f"Product Name: {row['Product Name']}, Description:{row['Description']}, Brand: {row['Brand']}, Price: {row['Base Price']} (sale: {row['Sale Price']}) {row['Currency']}, Categories: [{row['Primary Category']} - {row['Sub-category']}], Tags: {row['Tags']}, Sizes: {row['Size']}, Colors: {row['Color']}, Material: {row['Material']}, rating: {row['Average Rating']} ({row['Reviews']}), Vendor: {row['Vendor Name']}[{row['Vendor ID']}]", axis=1)
product_data_df['text_embedding'] = product_data_df.combined.apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))

product_data_df.to_json(r'data-embeddings.json', orient="records")

# getEmbeddingTime = datetime.now()
# print("Time to get embeddings:")
# print(getEmbeddingTime - starttime)

# get customer input and compare with product embeddings
# customer_input = "Hi! I need to buy a new winter jacket. It should be black."
# print("Customer input:")
# print(customer_input)
# response = openai.Embedding.create(
#     input=customer_input,
#     model="text-embedding-ada-002"
# )
# embeddings_customer_question = response['data'][0]['embedding']

# createEmbeddingTime = datetime.now()
# print("Time to create embeddings:")
# print(createEmbeddingTime - getEmbeddingTime)

# get top3 product recommendations
# product_data_df['search_products'] = product_data_df.text_embedding.apply(lambda x: cosine_similarity(x, embeddings_customer_question))
# product_data_df = product_data_df.sort_values('search_products', ascending=False)
# top_3_products_df = product_data_df.head(3)
# top_3_products_df

# productRecommendationTime = datetime.now()
# print("Time to get product recommendations:")
# print(productRecommendationTime - createEmbeddingTime)

# create message objects
# message_objects = []
# message_objects.append({"role":"system", "content":"You are an assistant that can recommend products from the available products dataset based on user questions."})
# message_objects.append({"role":"user", "content": customer_input})
# message_objects.append({"role":"user", "content": "Please give me a short explanation of your recommendations"})
# message_objects.append({"role":"user", "content": "Please talk to me like a person, don't just give me a list of recommendations"})
# message_objects.append({"role":"user", "content": "Please always include a Direct URL to marketplace for each recommendation"})
# message_objects.append({"role":"user", "content": f"Please always take into account the user's preferred brands: {user['preferred_brands']}"})
# message_objects.append({"role":"user", "content": f"Please always remove result for these vendors: {user['excluded_vendors']}"})
# message_objects.append({"role": "assistant", "content": "I found these 3 products I would recommend"})

# products_list = []
# for index, row in top_3_products_df.iterrows():
#     brand_dict = {'role': "assistant", "content": f"{row['combined']}"}
#     products_list.append(brand_dict)

# message_objects.extend(products_list)
# message_objects.append({"role": "assistant", "content":"Here's my summarized recommendation of products, and why it would suit you:"})

# completion = openai.ChatCompletion.create(
#   model="gpt-3.5-turbo",
#   messages=message_objects
# )
# print("Chatbot response:")
# print(completion.choices[0].message['content'])

# endtime = datetime.now()
# print("Time to get chatbot response:")
# print(endtime - productRecommendationTime)
# print("Total time:")
# print(endtime - starttime)
