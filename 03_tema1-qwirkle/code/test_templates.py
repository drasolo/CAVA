from config import settings
from templates import TemplateExtractor

def main():
    extractor = TemplateExtractor(
        settings["tiles_reference_image"],
        grid_size=settings["cell_count"],
    )

    print("[INFO] Calibrator loaded. Warped ref shape:", extractor.warped_ref.shape)
    print("[INFO] Check debug_templates/all_tiles_grid.png to verify grid.")

    # Extract templates
    shapes = extractor.extract_shapes()
    colors = extractor.extract_colors()

    print("\nExtracted shape templates:", shapes.keys())
    print("Extracted color templates:", colors.keys())

    print("\n[INFO] Check debug_templates/shapes and debug_templates/colors folders.")
    print("[INFO] If something looks wrong, adjust SHAPE_REF_CELLS or COLOR_REF_CELLS.")

if __name__ == "__main__":
    main()
