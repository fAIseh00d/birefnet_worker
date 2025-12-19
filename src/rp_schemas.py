INPUT_SCHEMA = {
    'filename': {
        'type': str,
        'required': True
    },
    'image_b64': {
        'type': str,
        'required': True
    },
    # Return only the mask instead of cutout
    'only_mask': {
        'type': bool,
        'required': False
    },
    # Background color as [R, G, B, A] (0-255 each)
    'bgcolor': {
        'type': list,
        'required': False
    },
}
