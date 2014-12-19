"""
A simple database for organizing parameters by two keys, an 
extension name and a parameter name. This is used for storing
model parameters, latent variables, and gpu pointers.
"""
import logging 
log = logging.getLogger("global_log")

class ParamsDatabase:
    def __init__(self):
        self.dbs = {}
        
    def addDatabase(self, name):
        """
        Add a database 
        """
        if name in self.dbs.keys():
            log.error("Database already exists with name: %s", name)
        else:
            self.dbs[name] = {}
        
    def __getitem__(*args):
        """
        Call this as paramsDatabase[db,key].
        args[0] = self
        args[1] = (db,key)
        """
        self = args[0]
        (db,key) = args[1]
        return self.dbs[db][key]
    
    def __setitem__(*args):
        """
        Call this as paramsDatabase[db,key].
        args[0] = self
        args[1] = (db,key,val)
        """
        self = args[0]
        (db,key) = args[1]
        val = args[2]
        
        self.dbs[db][key]=val 
        
    def __contains__(self, *args):
        """
        is key in db?
        """
        (db,key) = args[0]
        return (key in self.dbs[db].keys())