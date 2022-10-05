def export_as_gif(filename, images, frames_per_second=13, rubber_band=False):
    if rubber_band:
        images += images[2:-1][::-1]

    images[0].save(
        filename,
        save_all=True,
        append_images=images[1:],
        duration=1000 // frames_per_second,
        loop=0,
    )
