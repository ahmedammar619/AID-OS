#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Database configuration module for the AID-RL project.
Handles MySQL connection using SQLAlchemy.
"""

from collections import namedtuple
import random

# Define your data structures
Volunteer = namedtuple('Volunteer', ['volunteer_id', 'latitude', 'longitude', 'car_size'])
Recipient = namedtuple('Recipient', ['recipient_id', 'latitude', 'longitude', 'num_items'])

# 20 volunteers spread over a city grid
random.seed(42)
test_volunteers = [
    Volunteer(i,
        34.00  + random.uniform(-0.1,0.1),
        -118.25 + random.uniform(-0.1,0.1),
        random.choice([6, 8, 10, 12, 15, 20])
    )
    for i in range(1,21)
]

# 50 recipients: most need 1 box, a few outliers need 10–30
test_recipients = []
for i in range(1,51):
    lat = 34.00  + random.uniform(-0.15,0.15)
    lon = -118.25 + random.uniform(-0.15,0.15)
    if i % 20 == 0:
        boxes = random.choice([10,20,30])   # big outlier every 20th
    elif i % 7 == 0:
        boxes = random.choice([5,8,12])     # medium outlier every 7th
    else:
        boxes = 1                           # typical case
    test_recipients.append(Recipient(i, lat, lon, boxes))

# Quick sanity prints
# print("Volunteers:", len(test_volunteers))
# print("Recipients:", len(test_recipients))
# print(test_volunteers[:3], test_recipients[:5])



from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship
import os
import configparser
import pandas as pd

# Create a Base class for declarative class definitions
Base = declarative_base()

# Define ORM models for the database tables
class Volunteer(Base):
    __tablename__ = 'volunteer'
    
    volunteer_id = Column(Integer, primary_key=True)
    zip_code = Column(Integer)
    car_size = Column(Integer)
    longitude = Column(Float, nullable=True)
    latitude = Column(Float, nullable=True)
    replied = Column(String, nullable=False, default='No response')
    pickup_location_id = Column(Integer, ForeignKey('collect_location.location_id'))
    
    # Relationships
    pickup = relationship("Pickup", back_populates="volunteers")
    deliveries = relationship("Delivery", back_populates="volunteer")
    archived_deliveries = relationship("DeliveryArchive", back_populates="volunteer")
    
    def __repr__(self):
        return f"<Volunteer(volunteer_id={self.volunteer_id}, zip_code={self.zip_code}, car_size={self.car_size}, location=(32.7767, -96.7970))>"


class Recipient(Base):
    __tablename__ = 'recipient'
    
    recipient_id = Column(Integer, primary_key=True)
    latitude = Column(Float)
    longitude = Column(Float)
    num_items = Column(Integer)
    distributor_id = Column(Integer)
    replied = Column(String, nullable=False, default='No response')
    
    # Relationship with delivery archive
    deliveries = relationship("Delivery", back_populates="recipient")
    archived_deliveries = relationship("DeliveryArchive", back_populates="recipient")
    
    def __repr__(self):
        return f"<Recipient(recipient_id={self.recipient_id}, location=({self.latitude}, {self.longitude}), num_items={self.num_items})>"

class Pickup(Base):
    __tablename__ = 'collect_location'
    
    location_id = Column(Integer, primary_key=True, autoincrement=True)
    latitude = Column(Float)
    longitude = Column(Float)
    num_items = Column(Integer)
    active = Column(Integer, nullable=False, default=1)
    
    # Relationship with volunteers
    volunteers = relationship("Volunteer", back_populates="pickup")
    
    def __repr__(self):
        return f"<Pickup(location_id={self.location_id}, location=({self.latitude}, {self.longitude}), num_items={self.num_items})>"

class Delivery(Base):
    __tablename__ = 'deliveryTest'
    
    delivery_id = Column(Integer, primary_key=True, autoincrement=True)
    volunteer_id = Column(Integer, ForeignKey('volunteer.volunteer_id'))
    recipient_id = Column(Integer, ForeignKey('recipient.recipient_id'))
    status = Column(String, nullable=False, default='Pending')
    selected_date = Column(DateTime)
    
    # Define relationships
    volunteer = relationship("Volunteer", back_populates="deliveries")
    recipient = relationship("Recipient", back_populates="deliveries")
    
    def __repr__(self):
        return f"<Delivery(delivery_id={self.delivery_id}, volunteer_id={self.volunteer_id}, recipient_id={self.recipient_id})>"

class DeliveryArchive(Base):
    __tablename__ = 'delivery_archive'
    
    arch_id = Column(Integer, primary_key=True, autoincrement=True)
    volunteer_id = Column(Integer, ForeignKey('volunteer.volunteer_id'))
    recipient_id = Column(Integer, ForeignKey('recipient.recipient_id'))
    archive_date = Column(DateTime)
    
    # Define relationships
    volunteer = relationship("Volunteer", back_populates="archived_deliveries")
    recipient = relationship("Recipient", back_populates="archived_deliveries")
    
    def __repr__(self):
        return f"<DeliveryArchive(volunteer_id={self.volunteer_id}, recipient_id={self.recipient_id}, archive_date={self.archive_date})>"


class DatabaseHandler:
    """Handles database connections and queries for the AID-RL project."""
    
    def __init__(self, config_file=None):
        """Initialize the database handler with configuration."""
        if config_file and os.path.exists(config_file):
            self.config = self._load_config(config_file)
            self.engine = self._create_engine_from_config()
        else:
            # Default to local MySQL instance if no config is provided
            self.engine = create_engine('mysql+pymysql://root:@localhost/AID_OS')
        
        # Create a session factory
        self.Session = sessionmaker(bind=self.engine)
    
    def _load_config(self, config_file):
        """Load database configuration from a file."""
        config = configparser.ConfigParser()
        config.read(config_file)
        return config['DATABASE']
    
    def _create_engine_from_config(self):
        """Create a SQLAlchemy engine from configuration."""
        db_url = f"mysql+pymysql://{self.config['username']}:{self.config['password']}@{self.config['host']}/{self.config['database']}"
        return create_engine(db_url)
    
    def create_tables(self):
        """Create all tables defined in the Base metadata."""
        Base.metadata.create_all(self.engine)
    
    coordinates = {
        "75001": {"lat": 32.9576, "lon": -96.8389},  # Addison, Dallas County
        "75002": {"lat": 33.0969, "lon": -96.6144},  # Allen, Collin County
        "75003": {"lat": 32.9639, "lon": -96.7937},  # Dallas, Dallas County
        "75006": {"lat": 32.9756, "lon": -96.8917},  # Carrollton, Dallas County
        "75010": {"lat": 33.0304, "lon": -96.8855},  # Carrollton, Denton County
        "75013": {"lat": 33.1197, "lon": -96.6944},  # Allen, Collin County
        "75019": {"lat": 32.9753, "lon": -97.0023},  # Coppell, Dallas County
        "75023": {"lat": 33.0547, "lon": -96.7359},  # Plano, Collin County
        "75024": {"lat": 33.0759, "lon": -96.7843},  # Plano, Collin County
        "75033": {"lat": 33.1619, "lon": -96.7517},  # Frisco, Collin County
        "75038": {"lat": 32.8739, "lon": -96.9851},  # Irving, Dallas County
        "75039": {"lat": 32.8797, "lon": -96.9403},  # Irving, Dallas County
        "75040": {"lat": 32.9252, "lon": -96.6217},  # Garland, Dallas County
        "75041": {"lat": 32.8847, "lon": -96.6534},  # Garland, Dallas County
        "75042": {"lat": 32.9157, "lon": -96.6754},  # Garland, Dallas County
        "75044": {"lat": 32.9655, "lon": -96.6641},  # Garland, Dallas County
        "75048": {"lat": 32.9688, "lon": -96.5849},  # Sachse, Dallas County
        "75051": {"lat": 32.7219, "lon": -97.0003},  # Grand Prairie, Dallas County
        "75054": {"lat": 32.5959, "lon": -97.0439},  # Grand Prairie, Tarrant County
        "75063": {"lat": 32.9177, "lon": -97.0269},  # Irving, Dallas County
        "75070": {"lat": 33.1978, "lon": -96.6857},  # McKinney, Collin County
        "75071": {"lat": 33.2119, "lon": -96.6497},  # McKinney, Collin County
        "75074": {"lat": 33.0239, "lon": -96.6869},  # Plano, Collin County
        "75075": {"lat": 33.0219, "lon": -96.7389},  # Plano, Collin County
        "75080": {"lat": 32.9757, "lon": -96.7497},  # Richardson, Dallas County
        "75081": {"lat": 32.9489, "lon": -96.7117},  # Richardson, Dallas County
        "75082": {"lat": 32.9919, "lon": -96.6677},  # Richardson, Dallas County
        "75089": {"lat": 32.9389, "lon": -96.5477},  # Rowlett, Dallas County
        "75093": {"lat": 33.0359, "lon": -96.7897},  # Plano, Collin County
        "75094": {"lat": 33.0129, "lon": -96.6147},  # Plano, Collin County
        "75098": {"lat": 33.0139, "lon": -96.5347},  # Wylie, Collin County
        "75126": {"lat": 32.7459, "lon": -96.4607},  # Forney, Kaufman County
        "75150": {"lat": 32.8139, "lon": -96.6117},  # Mesquite, Dallas County
        "75166": {"lat": 33.0539, "lon": -96.4777},  # Lavon, Collin County
        "75212": {"lat": 32.7797, "lon": -96.8887},  # Dallas, Dallas County
        "75220": {"lat": 32.8677, "lon": -96.8637},  # Dallas, Dallas County
        "75227": {"lat": 32.7717, "lon": -96.6917},  # Dallas, Dallas County
        "75234": {"lat": 32.9277, "lon": -96.8517},  # Farmers Branch, Dallas County
        "75238": {"lat": 32.8767, "lon": -96.7097},  # Dallas, Dallas County
        "75243": {"lat": 32.9117, "lon": -96.7367},  # Dallas, Dallas County
        "75254": {"lat": 32.9477, "lon": -96.8017},  # Dallas, Dallas County
        "75287": {"lat": 33.0007, "lon": -96.8417},  # Dallas, Dallas County
        "75432": {"lat": 33.5839, "lon": -95.9337},  # Cooper, Delta County (likely outside DFW)
        "75454": {"lat": 33.2979, "lon": -96.5737},  # Melissa, Collin County
        "76002": {"lat": 32.6279, "lon": -97.0977},  # Arlington, Tarrant County
        "76006": {"lat": 32.7759, "lon": -97.0817},  # Arlington, Tarrant County
        "76010": {"lat": 32.7329, "lon": -97.0777},  # Arlington, Tarrant County
        "76039": {"lat": 32.6959, "lon": -97.0157},  # Euless, Tarrant County
        "76040": {"lat": 32.8239, "lon": -97.0207},  # Euless, Tarrant County
        "76051": {"lat": 32.9339, "lon": -97.0877},  # Grapevine, Tarrant County
        "76102": {"lat": 32.7559, "lon": -97.3297},  # Fort Worth, Tarrant County
        "76104": {"lat": 32.7289, "lon": -97.3217},  # Fort Worth, Tarrant County
        "76119": {"lat": 32.6919, "lon": -97.2707},  # Fort Worth, Tarrant County
        "76134": {"lat": 32.6479, "lon": -97.3287}   # Fort Worth, Tarrant County
    }

    def _get_lat_from_zip(self, zip_code):
        """Dummy function to convert zip code to latitude. Would be replaced with actual geocoding."""
        # In a real implementation, use a geocoding service or database
        return self.coordinates[str(zip_code)]["lat"] if str(zip_code) in self.coordinates else print(f"Invalid zip code: {zip_code}")
    
    def _get_lon_from_zip(self, zip_code):
        """Dummy function to convert zip code to longitude. Would be replaced with actual geocoding."""
        # In a real implementation, use a geocoding service or database
        return self.coordinates[str(zip_code)]["lon"]
    

    def get_all_volunteers(self):
        """Retrieve all volunteers from the database and convert zip codes to coordinates."""
        session = self.Session()
        volunteers = session.query(Volunteer).filter(
            (Volunteer.replied == 'Delivery') | (Volunteer.replied == 'Both')
        ).all()

        # Convert zip codes to coordinates and ensure car_size is integer
        for volunteer in volunteers:
            try:
                volunteer.car_size = int(volunteer.car_size)
                # Convert zip code to coordinates
                if str(volunteer.zip_code) in self.coordinates:
                    volunteer.longitude = self._get_lon_from_zip(volunteer.zip_code)
                    volunteer.latitude = self._get_lat_from_zip(volunteer.zip_code)
                else:
                    # Default to Dallas coordinates if zip not found
                    volunteer.longitude = 32.7767
                    volunteer.latitude = -96.7970
            except ValueError:
                volunteer.car_size = 0.001  # Handle invalid cases

        session.close()


        return volunteers
    
    def get_all_recipients(self):
        """Retrieve all recipients from the database."""
        session = self.Session()
        recipients = session.query(Recipient).filter(
            Recipient.replied == 'Yes',
            # Recipient.distributor_id == None
        ).all()
        session.close()
        return recipients
    
    def get_all_pickups(self):
        """Retrieve all pickup locations from the database."""
        session = self.Session()
        pickups = session.query(Pickup).filter(
            Pickup.active == 1  # Using 1 instead of True for MySQL
        ).all()
        session.close()
        return pickups


    def get_historical_deliveries(self):
        """Retrieve historical delivery data."""
        session = self.Session()
        result = session.query(DeliveryArchive).all()
        session.close()
        return result
    
    def save_assignment(self, volunteer_id, recipient_id):
        """Save a new volunteer-recipient assignment to the delivery table."""
        session = self.Session()
        
        new_delivery = Delivery(
            volunteer_id=volunteer_id,
            recipient_id=recipient_id,
            status='Confirmed',
            selected_date=pd.Timestamp.now()
        )
        
        session.add(new_delivery)
        session.commit()
        session.close()
    
    def bulk_save_assignments(self, assignments):
        """
        Save multiple volunteer-recipient assignments at once.
        
        Args:
            assignments: List of (volunteer_id, recipient_id) tuples
        """
        session = self.Session()
        
        # Create DeliveryArchive objects for each assignment
        delivery_objects = [
            Delivery(
                volunteer_id=vol_id,
                recipient_id=rec_id,
                status='Confirmed',
                selected_date=pd.Timestamp.now()
            )
            for vol_id, rec_id in assignments
        ]
        
        # Add all objects and commit
        session.add_all(delivery_objects)
        session.commit()
        session.close()
    
    def get_volunteer_historical_score(self, volunteer_id, recipient_id):
        """
        Calculate a historical match score for a volunteer-recipient pair
        based on previous successful deliveries.
        """
        session = self.Session()
        
        # Count previous successful matches
        count = session.query(DeliveryArchive).filter(
            DeliveryArchive.volunteer_id == volunteer_id,
            DeliveryArchive.recipient_id == recipient_id
        ).count()
        
        session.close()
        
        # Return a normalized score (0-3) based on the count
        if count > 3:
            return 3.0  # Maximum score
        return count * 1.0  # 1 point per previous match
        
    def execute_raw_query(self, query, params=None):
        """
        Execute a raw SQL query and return results.
        
        Args:
            query (str): The SQL query to execute
            params (tuple, optional): Parameters for the query
        
        Returns:
            list: List of dictionaries representing the query results
        """
        try:
            from sqlalchemy import text
            connection = self.engine.connect()
            
            # Create a text object with parameters
            if params:
                if isinstance(params, tuple) and len(params) == 1:
                    # Convert single-value tuple to dict with positional placeholders
                    sql = text(query.replace("%s", ":param0"))
                    result = connection.execute(sql, {"param0": params[0]})
                else:
                    # For multiple parameters or non-tuple params
                    sql = text(query)
                    result = connection.execute(sql, params)
            else:
                sql = text(query)
                result = connection.execute(sql)
                
            # Convert to list of dictionaries
            columns = result.keys()
            data = []
            for row in result:
                data.append(dict(zip(columns, row)))
                
            connection.close()
            return data
        except Exception as e:
            print(f"Error executing query: {e}")
            return []

#method to be added to a method to count the number of rows in an array
def count(array):
    print(len(array))
#method to be added to a method to print the first n rows of an array
def show(array, limit=5):
    for i in range(min(limit, len(array))):
        print(array[i])

# Example usage
if __name__ == "__main__":
    db = DatabaseHandler()
    # db.create_tables()
    Volunteers = db.get_all_volunteers()
    show(Volunteers)
    count(Volunteers)
    pickups = db.get_all_pickups()
    show(pickups)
    count(pickups)
    #draw the coords of the recipients on a graph and show the id of the point on hover
    # import matplotlib.pyplot as plt
    # plt.scatter([r.longitude for r in recipients], [r.latitude for r in recipients])
    # plt.show()

    # print("Database tables created successfully!")
