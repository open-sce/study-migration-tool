# Study Migration Tool

A Python application to assist in study migration analysis.

## Setup & Installation

By default, this application will attempt to load the example csv data provided with this repository. Please 
modify this path along with the other configuration options found in config.py as required.

#### Local

Create a virtual environment and run the `pip install -r requirements.txt` to install the required packages.

This application does not require any environment variables, however when running locally app.run_server does default to 
the following variables if set:

    host: str = os.getenv("HOST", "127.0.0.1"),
    port: int = os.getenv("PORT", "8050"),
    proxy: str = os.getenv("DASH_PROXY", None)

#### Cloud

See the following guides for deployment information suitable for this application:

[Azure - Python web app quickstart](https://learn.microsoft.com/en-us/azure/app-service/quickstart-python)

[AWS - Deploy python Flask](https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/create-deploy-python-flask.html)

#### Domino

Please see the [domino documentation](https://docs.dominodatalab.com/en/4.5/user_guide/de2589/publish-a-dash-app/) for 
more information on dash app publication with Domino.

## Usage

On startup, this application will load your source book of work data and generate a timeline overview of milestone 
activity following the default configuration specified in config.py.

These blocks of activity represent the milestone dates from your book of work plus/minus the offsets 
specified on the collapsable menu to the left of the screen. The timeframe and milestones located on this menu
can be modified at any time. 

The side menu also includes a "Hide Milestones Outside Timeframe" option which can be used to temporarily 
override the timeframe and return to a complete activity overview.

Below this main study activity view, additional search functionality is provided to generate study or compound specific 
timeline views as required.

All the graphs provide additional information on milestone activity when hovering alongside several different tools 
and utilities in a menu located in the top right corner. The legend on these graphs is also interactive, allowing 
you to quickly toggle the visibility of specific milestones on click.

## Contributing

We encourage you to contribute to the Study Migration Tool!

**Please do not make direct contributions to the OpenSCE repository! Instead, please visit the repository for this 
specific utility [here](https://github.com/Achieveintelligence/)**

This application is built using Dash which is a Python framework created by Plotly that uses Flask, Plotly.js and React 
under the hood. If you are unfamiliar with this framework, please see the [Dash documentation.](https://dash.plotly.com/)

To started contributing or report an issue, see our[Contribution Guidelines](/CONTRIBUTIONS.md).

## Upcoming Features

We're already busy working on a version 2 with lots of exciting new features so be sure to check back soon!

## Support

For support please contact the OpenSCE community.

## Lisense

The Study Migration Tool is released under the Apache 2.0 License.
