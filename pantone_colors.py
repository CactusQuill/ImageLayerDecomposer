"""
Helper module with Pantone color codes and a function to get all available color codes for display in the UI.
"""

def get_all_pantone_codes():
    """
    Returns a dictionary of all available Pantone color codes with their names.
    """
    pantone_colors = {
        # TPX Colors (Textile Paper eXtended)
        '19-4052 TCX': 'Classic Blue (2020)',
        '16-1546 TCX': 'Living Coral (2019)',
        '18-3838 TCX': 'Ultra Violet (2018)',
        '15-0343 TCX': 'Greenery (2017)',
        '13-1520 TCX': 'Rose Quartz (2016)',
        '14-4313 TCX': 'Serenity (2016)',
        '18-1438 TCX': 'Marsala (2015)',
        '17-1360 TCX': 'Tangerine Tango (2012)',
        '11-0601 TCX': 'Whisper White',
        '19-4005 TCX': 'Black',
        '19-1664 TCX': 'True Red',
        '17-1462 TCX': 'Flame Orange',
        '14-0756 TCX': 'Yellow Gold',
        '15-5534 TCX': 'Turquoise',
        '19-3950 TCX': 'Purple',
        '18-0135 TCX': 'Kelly Green',
        '14-4122 TCX': 'Sky Blue',
        
        # TPG Colors (Textile Paper Gloss)
        '19-4052 TPG': 'Classic Blue (2020)',
        '16-1546 TPG': 'Living Coral (2019)',
        '18-3838 TPG': 'Ultra Violet (2018)',
        '15-0343 TPG': 'Greenery (2017)',
        '13-1520 TPG': 'Rose Quartz (2016)',
        '14-4313 TPG': 'Serenity (2016)',
        '18-1438 TPG': 'Marsala (2015)',
        '17-1360 TPG': 'Tangerine Tango (2012)',
        '11-0601 TPG': 'Whisper White',
        '19-4005 TPG': 'Black',
        '19-1664 TPG': 'True Red',
        '17-1462 TPG': 'Flame Orange',
        '14-0756 TPG': 'Yellow Gold',
        '15-5534 TPG': 'Turquoise',
        '19-3950 TPG': 'Purple',
        '18-0135 TPG': 'Kelly Green',
        '14-4122 TPG': 'Sky Blue',
    }
    
    return pantone_colors
