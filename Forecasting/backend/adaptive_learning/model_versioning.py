"""
Model Versioning System

Manages model versions, tracks performance evolution, and enables rollback.
Uses semantic versioning: v{major}.{minor}.{patch}
"""

import io
import pickle
from datetime import datetime
from typing import Dict, Optional, List, Tuple
import torch
import torch.nn as nn
from pymongo import DESCENDING


class ModelVersionManager:
    """Manage model versions with semantic versioning"""
    
    def __init__(self, db):
        """
        Initialize version manager
        
        Args:
            db: Database instance
        """
        self.db = db
        self.versions_collection = db.db['model_versions']
        self._ensure_indexes()
    
    def _ensure_indexes(self):
        """Create database indexes for efficient queries"""
        self.versions_collection.create_index([
            ('symbol', 1),
            ('model_name', 1),
            ('version', -1)
        ])
        self.versions_collection.create_index([
            ('symbol', 1),
            ('model_name', 1),
            ('status', 1)
        ])
    
    def parse_version(self, version_str: str) -> Tuple[int, int, int]:
        """
        Parse version string to tuple
        
        Args:
            version_str: Version string (e.g., 'v1.2.3')
        
        Returns:
            (major, minor, patch) tuple
        """
        if version_str.startswith('v'):
            version_str = version_str[1:]
        
        parts = version_str.split('.')
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    
    def format_version(self, major: int, minor: int, patch: int) -> str:
        """Format version tuple to string"""
        return f"v{major}.{minor}.{patch}"
    
    def increment_version(self, current_version: str, update_type: str = 'patch') -> str:
        """
        Increment version number
        
        Args:
            current_version: Current version string
            update_type: 'major', 'minor', or 'patch'
        
        Returns:
            New version string
        """
        major, minor, patch = self.parse_version(current_version)
        
        if update_type == 'major':
            return self.format_version(major + 1, 0, 0)
        elif update_type == 'minor':
            return self.format_version(major, minor + 1, 0)
        else:  # patch
            return self.format_version(major, minor, patch + 1)
    
    def get_latest_version(self, symbol: str, model_name: str) -> str:
        """
        Get latest version for a model
        
        Args:
            symbol: Stock/Crypto symbol
            model_name: Name of the model
        
        Returns:
            Latest version string or 'v1.0.0' if none exists
        """
        doc = self.versions_collection.find_one(
            {'symbol': symbol, 'model_name': model_name},
            sort=[('trained_at', DESCENDING)]
        )
        
        return doc['version'] if doc else 'v1.0.0'
    
    def save_version(self, symbol: str, model_name: str, model: nn.Module,
                    scaler, config: Dict, metrics: Dict, 
                    update_type: str = 'patch',
                    training_data_range: Dict = None) -> str:
        """
        Save new model version
        
        Args:
            symbol: Stock/Crypto symbol
            model_name: Name of the model
            model: PyTorch model
            scaler: Data scaler
            config: Model configuration
            metrics: Performance metrics
            update_type: 'major', 'minor', or 'patch'
            training_data_range: {'start': date, 'end': date}
        
        Returns:
            New version string
        """
        # Determine new version
        current_version = self.get_latest_version(symbol, model_name)
        new_version = self.increment_version(current_version, update_type)
        
        # Serialize model
        model_buffer = io.BytesIO()
        torch.save(model.state_dict(), model_buffer)
        model_state = model_buffer.getvalue()
        
        # Serialize scaler
        scaler_state = pickle.dumps(scaler)
        
        # Create version document
        version_doc = {
            'symbol': symbol,
            'model_name': model_name,
            'version': new_version,
            'trained_at': datetime.utcnow(),
            'training_data_range': training_data_range or {},
            'model_state': model_state,
            'scaler_state': scaler_state,
            'config': config,
            'performance': metrics,
            'status': 'active',
            'update_type': update_type
        }
        
        # Deactivate previous active version
        self.versions_collection.update_many(
            {
                'symbol': symbol,
                'model_name': model_name,
                'status': 'active'
            },
            {'$set': {'status': 'archived'}}
        )
        
        # Insert new version
        self.versions_collection.insert_one(version_doc)
        
        # Archive old versions (keep last 10)
        self.archive_old_versions(symbol, model_name, keep=10)
        
        print(f"✅ Saved {model_name} {new_version} for {symbol}")
        return new_version
    
    def load_version(self, symbol: str, model_name: str, 
                    version: Optional[str] = None) -> Tuple[nn.Module, object, Dict]:
        """
        Load specific model version
        
        Args:
            symbol: Stock/Crypto symbol
            model_name: Name of the model
            version: Version string (None for latest active)
        
        Returns:
            (model, scaler, config) tuple
        """
        # Build query
        query = {'symbol': symbol, 'model_name': model_name}
        
        if version:
            query['version'] = version
        else:
            query['status'] = 'active'
        
        # Find version
        doc = self.versions_collection.find_one(
            query,
            sort=[('trained_at', DESCENDING)]
        )
        
        if not doc:
            raise ValueError(f"No version found for {symbol}/{model_name}")
        
        # Deserialize scaler
        scaler = pickle.loads(doc['scaler_state'])
        
        # Return model state, scaler, and config
        # Note: Model reconstruction happens in the caller
        return doc['model_state'], scaler, doc['config']
    
    def get_active_version(self, symbol: str, model_name: str) -> Optional[str]:
        """Get currently active version"""
        doc = self.versions_collection.find_one(
            {
                'symbol': symbol,
                'model_name': model_name,
                'status': 'active'
            }
        )
        return doc['version'] if doc else None
    
    def rollback(self, symbol: str, model_name: str, 
                version: Optional[str] = None) -> str:
        """
        Rollback to previous version
        
        Args:
            symbol: Stock/Crypto symbol
            model_name: Name of the model
            version: Specific version to rollback to (None for previous)
        
        Returns:
            Version string that was activated
        """
        if version is None:
            # Get previous version (second most recent)
            versions = list(self.versions_collection.find(
                {'symbol': symbol, 'model_name': model_name},
                sort=[('trained_at', DESCENDING)],
                limit=2
            ))
            
            if len(versions) < 2:
                raise ValueError("No previous version to rollback to")
            
            version = versions[1]['version']
        
        # Deactivate current
        self.versions_collection.update_many(
            {
                'symbol': symbol,
                'model_name': model_name,
                'status': 'active'
            },
            {'$set': {'status': 'archived'}}
        )
        
        # Activate target version
        result = self.versions_collection.update_one(
            {
                'symbol': symbol,
                'model_name': model_name,
                'version': version
            },
            {'$set': {'status': 'active'}}
        )
        
        if result.modified_count == 0:
            raise ValueError(f"Version {version} not found")
        
        print(f"🔄 Rolled back {model_name} to {version} for {symbol}")
        return version
    
    def archive_old_versions(self, symbol: str, model_name: str, keep: int = 10):
        """
        Archive old versions, keeping only the most recent N
        
        Args:
            symbol: Stock/Crypto symbol
            model_name: Name of the model
            keep: Number of versions to keep
        """
        # Get all versions sorted by date
        versions = list(self.versions_collection.find(
            {'symbol': symbol, 'model_name': model_name},
            sort=[('trained_at', DESCENDING)]
        ))
        
        # Mark old versions for deletion
        if len(versions) > keep:
            old_versions = versions[keep:]
            old_ids = [v['_id'] for v in old_versions]
            
            self.versions_collection.update_many(
                {'_id': {'$in': old_ids}},
                {'$set': {'status': 'deleted'}}
            )
            
            print(f"🗑️  Archived {len(old_ids)} old versions of {model_name}")
    
    def get_version_history(self, symbol: str, model_name: str, 
                           limit: int = 20) -> List[Dict]:
        """
        Get version history with performance metrics
        
        Args:
            symbol: Stock/Crypto symbol
            model_name: Name of the model
            limit: Maximum number of versions to return
        
        Returns:
            List of version documents
        """
        versions = list(self.versions_collection.find(
            {
                'symbol': symbol,
                'model_name': model_name,
                'status': {'$ne': 'deleted'}
            },
            {
                'version': 1,
                'trained_at': 1,
                'performance': 1,
                'status': 1,
                'update_type': 1
            },
            sort=[('trained_at', DESCENDING)],
            limit=limit
        ))
        
        return versions
    
    def compare_versions(self, symbol: str, model_name: str,
                        version1: str, version2: str) -> Dict:
        """
        Compare performance between two versions
        
        Args:
            symbol: Stock/Crypto symbol
            model_name: Name of the model
            version1: First version
            version2: Second version
        
        Returns:
            Comparison dictionary
        """
        v1_doc = self.versions_collection.find_one({
            'symbol': symbol,
            'model_name': model_name,
            'version': version1
        })
        
        v2_doc = self.versions_collection.find_one({
            'symbol': symbol,
            'model_name': model_name,
            'version': version2
        })
        
        if not v1_doc or not v2_doc:
            raise ValueError("One or both versions not found")
        
        return {
            'version1': {
                'version': version1,
                'trained_at': v1_doc['trained_at'],
                'performance': v1_doc['performance']
            },
            'version2': {
                'version': version2,
                'trained_at': v2_doc['trained_at'],
                'performance': v2_doc['performance']
            },
            'improvement': {
                'rmse': v1_doc['performance']['rmse'] - v2_doc['performance']['rmse'],
                'mae': v1_doc['performance']['mae'] - v2_doc['performance']['mae'],
                'mape': v1_doc['performance']['mape'] - v2_doc['performance']['mape']
            }
        }
