import plotly.graph_objects as go


def age_to_transported(df):
    age_transported_counts = df.groupby('Age')['Transported'].sum().reset_index()
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=age_transported_counts['Age'],
            y=age_transported_counts['Transported'],
            name='Transported'
        )
    )

    age_not_transported_counts = df.groupby('Age')['Transported'].count() - age_transported_counts['Transported']
    fig.add_trace(
        go.Bar(
            x=age_transported_counts['Age'],
            y=age_not_transported_counts,
            name='Not Transported'
        )
    )
    fig.update_layout(
        xaxis_title='Age',
        yaxis_title='Number of Individuals',
        title='Transported and Not Transported Individuals by Age',
        template="plotly_dark"  # Add this line for a black theme
    )

    fig.show()
