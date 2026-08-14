# HomeXperia - API Backend

A Flask-based image processing API that enables real-time interior design visualization. It allows users to virtually apply textures — curtains, rugs, flooring, and wall patterns — onto room images using AI-powered segmentation masks.

---

## Features

- **Room Image Processing** — Apply product textures (curtain, rug, floor, wall) onto room images using layered rendering
- **AI Mask Generation** — Integrates with a SAM (Segment Anything Model) API to auto-generate segmentation masks from click coordinates
- **Mask Caching** — Generated masks are cached locally to avoid redundant API calls
- **Layer Stack System** — Supports multi-layer rendering with history, re-applying previously applied hotspots in order
- **Image Upscaling & Preprocessing** — Auto-upscales and sharpens room images before texture application for high-quality output
- **PDF Report Generation** — Generates downloadable design summary PDFs
- **Admin Utilities** — Cleanup endpoints for managing generated files, uploads, and masks
- **QR Code Generator** — Utility script to generate QR codes for customer verification links
- **API Key Authentication** — All endpoints protected via `x-api-key` header; admin endpoints require an additional `Authorization` token

---

## Tech Stack

| Layer | Technology |

| Framework | Flask |
| Image Processing | OpenCV (`opencv-python-headless`), NumPy |
| AI Segmentation | SAM API (external, via HTTP) |
| PDF Generation | FPDF |
| Background Tasks | APScheduler |
| Database | Supabase |
| Server | Gunicorn |
| Config | python-dotenv |

---

## Project Structure

```
homexperia-backend-api/
├── app.py                  # Main Flask application & all API routes
├── generate_apikey.py      # Utility to generate API keys and tokens
├── generate_qrcode.py      # Utility to generate QR codes
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (not committed)
├── data/
│   └── rooms_data.json     # Room metadata store
├── static/
│   └── room-images/        # Static room images
├── utils/
│   ├── curtain.py          # Curtain texture application logic
│   ├── rugs.py             # Rug texture application logic
│   ├── floor.py            # Floor texture application logic
│   ├── wall.py             # Wall texture application logic
│   └── pdf_generator.py    # PDF report generation
├── uploads/                # Uploaded room images (auto-created)
├── generated/              # Output processed images (auto-created)
├── masks/                  # Cached segmentation masks (auto-created)
└── Debugs/                 # Debug images for preprocessing steps
```

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/skyai-dev/homexperia-backend-api.git
cd homexperia-backend-api
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
APP_API_KEY=your_api_key_here
AUTH_TOKEN=Bearer your_admin_auth_token_here
SAM_API_URL=https://your-sam-api-endpoint.com/segment
SAM_API_KEY=your_sam_api_key_here
```

To generate a new API key, run:

```bash
python generate_apikey.py
```

---

## Running the Server

### Production (Gunicorn)

```bash
gunicorn --bind 0.0.0.0:3000 app:app
```


## API Endpoints

All endpoints require the header:
```
x-api-key: <your_api_key>
```

---

### `POST /api/upload`

Upload a room image via URL or Base64.

**Request body:**
```json
{
  "imageUrl": "https://example.com/room.jpg"
}
```
or
```json
{
  "imageBase64": "data:image/jpeg;base64,/9j/..."
}
```

**Response:**
```json
{
  "success": true,
  "localFilePath": "uploads/upload_abc123.jpg",
  "serverBaseUrl": "http://your-server/uploads/upload_abc123.jpg"
}
```

---

### `POST /api/process-room`

Apply one or more product textures onto a room image. Supports layered history re-rendering.

**Request body:**
```json
{
  "roomId": "room_001",
  "baseImageUrl": "https://example.com/room.jpg",
  "applyHotspot": [
    {
      "hotspotId": "hs_floor_1",
      "category": "floor",
      "coords": { "x": 540, "y": 720 },
      "product": { "productImageUrl": "https://example.com/tile.jpg" },
      "mask_image": "https://example.com/mask.png",
      "settings": {
        "repeat": 12,
        "rotation": 0,
        "shading": 0.6,
        "groutWidth": 2,
        "groutColor": "#cccccc"
      }
    }
  ],
  "appliedHotspots": [],
  "remainingHotspots": []
}
```

**Response:**
```json
{
  "success": true,
  "finalImageUrl": "http://your-server/generated/final_room_001_1712345678.jpg",
  "appliedHotspots": [...],
  "remainingHotspots": []
}
```

---

### `POST /api/mask-generation`

Generate a segmentation mask for a specific point in a room image.

**Request body:**
```json
{
  "roomId": "room_001",
  "baseImageUrl": "https://example.com/room.jpg",
  "coords": { "x": 540, "y": 720 }
}
```

**Response:**
```json
{
  "success": true,
  "roomId": "room_001",
  "maskImageUrl": "http://your-server/masks/mask_room_001_hotspotId.png"
}
```

---

### `POST /api/reset`

Delete all cached masks and generated images for a room.

**Request body:**
```json
{
  "roomId": "room_001"
}
```

---

### `POST /api/generate-pdf`

Generate a PDF design summary report for a room.

**Request body:** Room and product data (JSON). Returns a downloadable PDF file.

---

### `GET /api/rooms`

Get a list of all available rooms with their image URLs.

---

### `GET /api/room/<room_id>`

Get metadata and image URL for a specific room.

---

### `DELETE /api/admin/cleanup/<target_folder>`

**Admin only** — requires both `x-api-key` and `Authorization` headers.

Delete all files in a specified folder. `target_folder` must be one of: `uploads`, `generated`, `masks`.

---

## Supported Texture Categories

| Category | Keyword Match | Settings |
|---|---|---|
| Curtain | `curtain` | `repeat`, `shading` |
| Rug | `rug` | `rotation` |
| Floor | `floor` | `repeat`, `rotation`, `groutWidth`, `groutColor` |
| Wall | `wall` | `repeat` |

---

## Utilities

### Generate API Key

```bash
python generate_apikey.py
```

Outputs a hex token, SHA-256 hash, and URL-safe token for use as `APP_API_KEY` or `AUTH_TOKEN`.

### Generate QR Code

```bash
python generate_qrcode.py
```

Generates a QR code PNG for a customer verification URL. Edit the `qr_data` variable in the script before running.

---

## Environment Variables Reference

| Variable | Description |
|---|---|
| `APP_API_KEY` | API key required in `x-api-key` header for all endpoints |
| `AUTH_TOKEN` | Bearer token required for admin endpoints |
| `SAM_API_URL` | Endpoint URL for the SAM segmentation API |
| `SAM_API_KEY` | API key for the SAM segmentation service |