import plotly.graph_objects as go


def side_to_h_planet(df):
    df["side"] = df["side"].map({"P": 0, "S": 1})
    age_transported_counts = df.groupby('HomePlanet')['side'].sum().reset_index()
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=age_transported_counts['HomePlanet'],
            y=age_transported_counts['side'],
            name='side 1'
        )
    )

    age_not_transported_counts = df.groupby('HomePlanet')['side'].count()
    fig.add_trace(
        go.Bar(
            x=age_transported_counts['HomePlanet'],
            y=age_not_transported_counts,
            name='side 0'
        )
    )
    fig.update_layout(
        xaxis_title='home planet',
        yaxis_title='side count',
        title='planet to side',
        template="plotly_dark"  # Add this line for a black theme
    )

    fig.show()


def plot_count(data):
    # Group the filtered data by 'HomePlanet' and count the number of people from each planet
    planet_counts = data['Age'].value_counts()

    # Create a bar chart
    fig = go.Figure(go.Bar(
        x=planet_counts.index,
        y=planet_counts.values,
    ))

    # Customize the layout
    fig.update_layout(
        title='Distribution of People from Each Planet',
        xaxis_title='Planet',
        yaxis_title='Number of People',
        template="plotly_dark"
    )

    # Show the plot
    fig.show()
