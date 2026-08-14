import qrcode

qr_data = "http://163.227.92.128/verify?customer-code=0007668N1ZE"

qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_H,
    box_size=10,
    border=4,
)
qr.add_data(qr_data)
qr.make(fit=True)

img = qr.make_image(fill_color="black", back_color="white").convert("RGBA")

img.save("Austhavinayak_QR.png")