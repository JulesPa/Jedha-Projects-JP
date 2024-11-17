import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np


data = pd.read_excel('/get_around_delay_analysis.xlsx')

st.title("GetAround Rental Delay Analysis")
st.write("""
This dashboard helps analyze the impact of rental delays on the GetAround platform. It provides insights to guide decisions
on minimum delay thresholds.
""")

st.subheader("Dataset Preview")
st.dataframe(data.head())  

st.subheader("Feature Descriptions")
st.write("""
- **rental_id**: Unique identifier for each rental.
- **delay_at_checkout_in_minutes**: Delay in minutes for returning the car at checkout.
- **checkin_type**: The type of check-in, which could be:
  - **Mobile**: Driver and owner meet and sign the rental agreement on the owner’s smartphone.
  - **Connect**: Driver unlocks the car using their smartphone, without meeting the owner.
- **state**: Status of the rental, indicating if it was completed, canceled, or involved any issues.
- **other relevant features**: Additional features as per dataset specifics, which may include car type, mileage, and other booking details.
""")

st.header("Late Checkout Analysis")

data['delay_at_checkout_in_minutes'].fillna(0, inplace=True)
late_checkouts = data[data['delay_at_checkout_in_minutes'] > 0]
late_checkout_pct = (len(late_checkouts) / len(data)) * 100

fig = px.histogram(
    data['delay_at_checkout_in_minutes']/60,
    x='delay_at_checkout_in_minutes',
    nbins=50000,
    title="Number of late checkouts",
    labels={'delay_at_checkout_in_minutes': 'Delay at Checkout (Hours)'},
    log_y=True 
)

fig.update_xaxes(range=[0, 6])
st.plotly_chart(fig)

st.write(f"**Percentage of Late Checkouts**: {late_checkout_pct:.2f}%")
st.write("This percentage represents rentals returned late, potentially impacting the next rental. Assuming that all\
         'NaN' values for 'delay_at_checkout_in_minutes' represent a checkout a the expected hour.")

st.header("Canceled Reservations Overview")
total_count = data['state'].notna().sum()  
canceled_count = (data['state'] == 'canceled').sum()
canceled_percentage = (canceled_count / total_count) * 100
z_score = 1.96  
p = canceled_count / total_count  
n = total_count    
canceled_margin_of_error = z_score * np.sqrt((p * (1 - p)) / n) * 100  


previous = data[data['previous_ended_rental_id'].notna()]
previous_canceled = previous[previous['state'] == 'canceled']
previous_canceled.drop(columns=['delay_at_checkout_in_minutes'], inplace=True)
previous_canceled = previous_canceled.merge(
    data[['rental_id', 'delay_at_checkout_in_minutes']],
    left_on='previous_ended_rental_id',
    right_on='rental_id',
    how='left'
)
previous_canceled.rename(columns={'delay_at_checkout_in_minutes': 'previous_delay_at_checkout'}, inplace=True)
delay_canceled = previous_canceled[previous_canceled['previous_delay_at_checkout'].notna()]
impact = previous_canceled[previous_canceled['previous_delay_at_checkout']>previous_canceled['time_delta_with_previous_rental_in_minutes']]

total_previous = previous['state'].notna().sum()
previous_canceled_count = (previous['state'] == 'canceled').sum()
previous_canceled_percentage = (previous_canceled_count / total_previous ) * 100

p = previous_canceled_count / total_previous  
n = previous_canceled_count  
previous_margin_of_error = z_score * np.sqrt((p * (1 - p)) / n) * 100  

delay = data[data['time_delta_with_previous_rental_in_minutes'].notna()]
total_delay = delay['state'].notna().sum()  
delay_canceled_count = (delay['state'] == 'canceled').sum()
delay_canceled_percentage = (delay_canceled_count / total_delay) * 100

p = delay_canceled_count / total_delay  
n = total_delay  
delay_margin_of_error = z_score * np.sqrt((p * (1 - p)) / n) * 100 
 
impact_count = (impact['state'] == 'canceled').sum()
impact_percentage = (impact_count / total_previous) * 100

z_score = 1.96  
p = impact_count / total_previous  
n = total_previous 
impact_margin_of_error = z_score * np.sqrt((p * (1 - p)) / n) * 100  

st.markdown(f"""
<div style="text-align: justify;">
    From those datas we can conclude that canceled reservations account for {canceled_percentage:.2f} ± {canceled_margin_of_error:.2f}%\
     of all reservations. Obviously, canceled reserevations are not all due to late checkouts. To proceed with a more precisize analysis,\
     we will need to focus on the datas that give us inforamtions about the previous reservations and about the time diference between\
     the current and the previous reservation. Unfortunally, only a small portion of the data provide those informations. Noneless, analysis\
     shows that for the subset providing insights about the previous reservation, canceled reservations account for {previous_canceled_percentage:.2f} ± {previous_margin_of_error:.2f}% of all reservations.\
     The same analysis shows that canceled reservations account for {delay_canceled_percentage:.2f} ± {delay_margin_of_error:.2f}% of all reservations in\
     the subset providing insights about the time difference between the current and the previous reservation (note that the values being the same\
     is coincidence, both those subsets are not the same). Those valuebeing really close to the one for the whole dataset both those subsets do not seems biased on the regard.\
     It seems safe to draw conclusion from those subset of data for the whole dataset. That being said, if we look at canceled reservations where the time difference between\
     two consecutive reservations is lower than delay at which the previous reservation was returned, we can see that they account for\
     {impact_percentage:.2f} ± {impact_margin_of_error:.2f}% of all reservations. We can estimate that this value represent a good estimation of the company loss\
     due to late checkouts eventhough that value could be a slighly overestimated (some of those reservations might have been canceled regardless).
</div>
""",unsafe_allow_html=True)




st.header("Impact of Delay Threshold and Car Type on Rentals")

average_time_delta = data['time_delta_with_previous_rental_in_minutes'].mean()
st.markdown(f"""
<div style="text-align: justify;">
    To try to have the best understanding of what impact the set up of a minimum delay between to rentals policy could have on the company revenue,\
     we will look at the number of rentals affected by different delay thresholds and car types. Firts we only look at the impact using the datas that provide\
     imformation regarding the delay with the previous reservation. But to try to be more precise, we will also look at the impact using all datas. For that we will use\
     the average time difference between two consecutive rentals ({average_time_delta:.2f} min) as a proxy for the delay with the previous reservation.
</div>
""",unsafe_allow_html=True)

car_type = st.selectbox("Select check-in Type", options=["All", "Connect", "Mobile"])
threshold = st.slider("Select a minimum delay threshold (minutes)", 0, 120, 30)


if car_type == "Connect":
    filtered_data = previous[(previous['time_delta_with_previous_rental_in_minutes'] < threshold) & (data['checkin_type'] == 'connect')]
elif car_type == "Mobile":
    filtered_data = previous[(previous['time_delta_with_previous_rental_in_minutes'] < threshold) & (data['checkin_type'] == 'mobile')]
else:
    filtered_data = previous[(previous['time_delta_with_previous_rental_in_minutes'] < threshold)]

affected_rentals = len(filtered_data)
affected_rentals_pct = ((affected_rentals) /len(previous)) * 100

st.subheader("First case : looking only at delay with the previous reservation")
st.write(f"**Number of rentals affected by threshold**: {affected_rentals}")
st.write(f"**Percentage of total rentals affected**: {affected_rentals_pct:.2f}%")

if car_type == "Connect":
    filtered_data_2 = data[(data['delay_at_checkout_in_minutes'] > (average_time_delta -threshold)) & (data['checkin_type'] == 'connect')]
elif car_type == "Mobile":
    filtered_data_2 = data[(data['delay_at_checkout_in_minutes'] > (average_time_delta -threshold)) & (data['checkin_type'] == 'mobile')]
else:
    filtered_data_2 = data[data['delay_at_checkout_in_minutes'] > (average_time_delta -threshold)]


affected_rentals_2 = len(filtered_data_2)
affected_rentals_pct_2 = ((affected_rentals_2) /len(data)) * 100

st.subheader("Second case : using the average time difference between two consecutive rentals as a proxy for all datas")
st.write(f"**Number of rentals affected by threshold**: {affected_rentals_2}")
st.write(f"**Percentage of total Rentals affected**: {affected_rentals_pct_2:.2f}%")


st.markdown("""
<div style="text-align: justify;">
    Looking at the two cases above make draw two different conclusions. In the first case we can see that the amount of reservations made without a time delay with\
     the previous one is so important that it exeeds the 2% of cancelled reservations due to a late checkout. therefore setting up a delay could be detrimental to the company.\
     To be completly conclusive we would have to know what amount of reservations would have been canceled if there was a (short) delay. In the second case we can see that
     the conclusion is not the same, setting up a delay only impact reservations by more than 2% only if the delay start to exceed 75 min (note that impacted reservations \
     do not start at 0 in this case because their a part of late checkouts that exceeds the average time difference between two consecutive rentals). If we could have more confidence \
     regarding the average time difference between two consecutive rentals we could have made a better conclusion. Note that those does not seemed to be impacted by the check-in type.
</div>
""",unsafe_allow_html=True)