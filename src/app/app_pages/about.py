# -*- coding: utf-8 -*-
# """
# about.py
# Created on Dec 17, 2024
# @ Author: Mazhar
# """

import streamlit as st


def display_about_text() -> None:
    st.markdown(
        """
    ## Omdena-IPAGE Project

    This project is developing an AI-driven predictive system to assess soil nutrients, specifically micro-nutrients like boron and zinc, as well as soil organic carbon, enhancing agricultural productivity and optimizing fertilizer use for smallholder farmers in Bangladesh.

    ### The Problem

    Smallholder farmers face significant challenges in achieving optimal agricultural productivity due to the lack of access to precise soil nutrient data, especially concerning micro-nutrients like boron and zinc, as well as soil organic carbon (SOC). Current soil testing technologies primarily focus on macronutrients such as Nitrogen, Phosphorus, and Potassium, leaving critical gaps in data for micro-nutrients and SOC. This limitation hampers the ability of farmers to apply the correct fertilizer balance, which is essential for maximizing crop yields and maintaining soil health.

    ### Impact of the Problem:

    *   **Suboptimal Crop Yields:** Without precise data on essential micro-nutrients and SOC, farmers cannot tailor their fertilizer use to the specific needs of their soil. This often leads to suboptimal crop yields, as plants may not receive the necessary nutrients in the correct proportions to thrive.

    *   **Environmental Harm:** The absence of detailed nutrient data can lead to over- or under-fertilization. Over-fertilization not only wastes resources and increases costs for farmers but also contributes to environmental issues such as water pollution and greenhouse gas emissions from excess nitrogen. Under-fertilization, on the other hand, can lead to soil degradation and reduced soil fertility over time, further diminishing the productive capacity of the land.

    *   **Economic Inefficiency:** Inaccurate fertilizer application can result in economic inefficiencies, where farmers spend more on inputs without seeing proportional benefits in crop output. This situation can strain the financial stability of smallholder farmers, who typically operate on thin profit margins.

    *   **Barrier to Sustainable Agriculture Practices:** Lack of comprehensive soil health data impedes the adoption of sustainable agricultural practices. Accurate information about soil nutrients, including micro-nutrients and organic carbon, is crucial for implementing practices that enhance soil health and long-term agricultural sustainability.

    In response to these challenges, this Omdena-IPAGE challenge aims to develop a solution to leverage existing soil data to predict missing elements such as SOC, boron, and zinc, and to provide comprehensive fertilizer recommendations. This project aims to enhance the current portable soil testing devices and integrate predictive analytics capabilities into them. By doing so, the goal is to generate a more complete picture of soil health for smallholder farmers, enabling them to optimize fertilizer application, enhance crop yields, reduce environmental impact, and improve economic efficiency in their farming operations. The project seeks to bridge the data gap and support sustainable farming practices, contributing significantly to the economic well-being and environmental health of agricultural communities.

    ### The Goals

    The ultimate objective of this project is to develop and deploy an advanced AI-driven system to predict missing soil nutrients, specifically micro-nutrients like boron and zinc, as well as soil organic carbon (SOC), for smallholder farmers in Bangladesh. This initiative will involve the integration of existing soil sensor data, machine learning model development, and chemical validation to create a system that not only predicts soil nutrient levels but also provides actionable insights for optimal fertilizer use. The project will unfold over several key milestones, each planned to ensure the successful development and deployment of this transformative technology:

    *   **Project Setup and Data Integration:** Finalize project scope, gather existing soil sensor data, and set up the necessary infrastructure. This initial phase lays the foundation for the entire project by establishing clear objectives and securing the data needed for model training.

    *   **Initial Model Development:** Develop initial machine learning models to predict levels of SOC, boron, and zinc. Conduct preliminary tests to assess the accuracy of these models, focusing on their ability to handle the complexity of soil data and nutrient interactions.

    *   **Model Refinement and Validation:** Refine the machine learning models based on initial test feedback and conduct chemical tests for further validation. This phase ensures the models are robust and reliable, accurately reflecting the nutrient status of the soils.

    *   **Finalization and Reporting:** Finalize the models and prepare the Testing and Validation Report. This report will document the methodologies, results, and the accuracy of the models in predicting soil nutrients.

    *   **Delivery and Future Planning:** Deliver the Proof of Concept (PoC) and complete documentation of the project. Begin planning for scaling the solution to other regions, adapting the models to accommodate different soil types and agricultural practices.

    Thus, this project aims to deliver a sophisticated AI-driven solution that revolutionizes soil nutrient prediction for smallholder farmers. By providing a more efficient, accurate, and scalable system, this initiative promises substantial benefits in improving agricultural productivity, reducing environmental impact from fertilizer misuse, and enhancing economic outcomes for farmers. This strategic approach ultimately contributes to more informed agricultural practices and improved food security in targeted regions.
    
    
    Email: mazqoty.01@gmail.com
    
    """
    )
