#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Database configuration module for the AID-RL project.
Handles MySQL connection using SQLAlchemy.
"""

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
    replied = Column(String, nullable=False, default='No response')
    
    # Relationship with delivery archive
    deliveries = relationship("Delivery", back_populates="volunteer")
    archived_deliveries = relationship("DeliveryArchive", back_populates="volunteer")
    
    def __repr__(self):
        return f"<Volunteer(volunteer_id={self.volunteer_id}, zip_code={self.zip_code}, car_size={self.car_size})>"


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
            self.engine = create_engine('mysql+pymysql://root:@localhost/AID_RL')
        
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
    
    def get_all_volunteers(self):
        """Retrieve all volunteers from the database."""
        session = self.Session()
        volunteers = session.query(Volunteer).filter(
            (Volunteer.replied == 'Delivery') | (Volunteer.replied == 'Both')
        ).all()

        # Ensure car_size is converted to an integer
        for volunteer in volunteers:
            try:
                volunteer.car_size = int(volunteer.car_size)
            except ValueError:
                volunteer.car_size = 0.001  # Handle invalid cases

        session.close()
        return volunteers
    
    def get_all_recipients(self):
        """Retrieve all recipients from the database."""
        session = self.Session()
        recipients = session.query(Recipient).filter(
            # Recipient.replied == 'Yes',
            Recipient.distributor_id == None
        ).all()
        session.close()
        return recipients


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
    recipients = db.get_all_recipients()
    show(recipients)
    count(recipients)
    #draw the coords of the recipients on a graph and show the id of the point on hover
    import matplotlib.pyplot as plt
    plt.scatter([r.longitude for r in recipients], [r.latitude for r in recipients])
    plt.show()

    # print("Database tables created successfully!")
