import streamlit as st
from pandas.io.formats.style import Styler

def display_styled_data_table(
    data, 
    selected_columns: list, 
    cfg, 
    highlight_ideal_values: callable
) -> None:
    """
    Display a DataFrame with styled values based on ideal ranges.

    Args:
        data (pd.DataFrame): Input DataFrame to be displayed.
        selected_columns (list): List of selected columns for styling.
        cfg: Configuration object containing style and range settings.
        highlight_ideal_values (callable): Function to highlight values based on a range.
    """
    # Define ideal ranges from config
    ideal_ranges: dict = {
        cfg.STAPP.STATISTICS_CARD.COL_1: tuple(cfg.STAPP["IDEAL_RANGES"]["pH"]),
        cfg.STAPP.STATISTICS_CARD.COL_2: tuple(cfg.STAPP["IDEAL_RANGES"]["SOC_PERCENT"]),
        cfg.STAPP.STATISTICS_CARD.COL_3: tuple(cfg.STAPP["IDEAL_RANGES"]["BORON_B_UG_G"]),
        cfg.STAPP.STATISTICS_CARD.COL_4: tuple(cfg.STAPP["IDEAL_RANGES"]["ZINC_ZN_UG_G"]),
    }

    # Display ideal ranges in expander
    with st.expander("View Ideal Ranges"):
        st.write(
            f"""
            - **pH**: {ideal_ranges[cfg.STAPP.STATISTICS_CARD.COL_1][0]} to {ideal_ranges[cfg.STAPP.STATISTICS_CARD.COL_1][1]}  
            - **SOC (%)**: {ideal_ranges[cfg.STAPP.STATISTICS_CARD.COL_2][0]} to {ideal_ranges[cfg.STAPP.STATISTICS_CARD.COL_2][1]}  
            - **Boron (B) (ug/g)**: {ideal_ranges[cfg.STAPP.STATISTICS_CARD.COL_3][0]} to {ideal_ranges[cfg.STAPP.STATISTICS_CARD.COL_3][1]}  
            - **Zinc (Zn) (ug/g)**: {ideal_ranges[cfg.STAPP.STATISTICS_CARD.COL_4][0]} to {ideal_ranges[cfg.STAPP.STATISTICS_CARD.COL_4][1]}  
            """
        )

    # Background color for highlighting
    color: str = cfg.STAPP["STYLES"]["COL_BACKGROUND_COLOR"]

    # Initialize Styler for selected columns
    styled_data: Styler = data[selected_columns].style

    # Apply highlighting for each column dynamically
    for col_key, ideal_range in ideal_ranges.items():
        if col_key in selected_columns:
            styled_data = styled_data.apply(
                lambda col: [
                    highlight_ideal_values(val, *ideal_range, color=color)
                    for val in col
                ],
                subset=[col_key],
            )

    # Display the styled DataFrame
    st.write(styled_data)
