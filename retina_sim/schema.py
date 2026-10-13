"""Dependency-free validator for the JSON Schema keywords used by our schema.

This validates this project's generated schema subset, not arbitrary draft-2020-12
schemas. Provenance semantics still require Registry validation.
"""
import json,re
from .contracts import ContractError

def validate_schema(value,schema,path='$'):
    types={'object':lambda v:isinstance(v,dict),'array':lambda v:isinstance(v,(list,tuple)),
           'string':lambda v:isinstance(v,str),'number':lambda v:isinstance(v,(int,float)) and not isinstance(v,bool),
           'integer':lambda v:isinstance(v,int) and not isinstance(v,bool),'boolean':lambda v:isinstance(v,bool),'null':lambda v:v is None}
    if 'type' in schema:
        expected=schema['type']; expected=[expected] if isinstance(expected,str) else expected
        if not any(types[t](value) for t in expected): raise ContractError(path+': schema type mismatch')
    if 'enum' in schema and value not in schema['enum']: raise ContractError(path+': enum mismatch')
    if 'const' in schema and value!=schema['const']: raise ContractError(path+': const mismatch')
    if isinstance(value,str):
        if len(value)<schema.get('minLength',0): raise ContractError(path+': empty string')
        if 'pattern' in schema and not re.search(schema['pattern'],value): raise ContractError(path+': pattern mismatch')
    if isinstance(value,(int,float)) and not isinstance(value,bool) and 'minimum' in schema and value<schema['minimum']:
        raise ContractError(path+': minimum')
    if isinstance(value,dict):
        if any(k not in value for k in schema.get('required',[])): raise ContractError(path+': required field missing')
        properties=schema.get('properties',{})
        if schema.get('additionalProperties') is False and set(value)-set(properties): raise ContractError(path+': unknown fields')
        for k,v in value.items():
            if k in properties: validate_schema(v,properties[k],path+'.'+k)
    if isinstance(value,(list,tuple)):
        if len(value)<schema.get('minItems',0): raise ContractError(path+': minItems')
        if schema.get('uniqueItems') and len({json.dumps(x,sort_keys=True) for x in value})!=len(value): raise ContractError(path+': duplicate items')
        if 'items' in schema:
            for i,v in enumerate(value): validate_schema(v,schema['items'],path+f'[{i}]')
    for sub in schema.get('allOf',[]): validate_schema(value,sub,path)
    if 'if' in schema:
        try: validate_schema(value,schema['if'],path)
        except ContractError:
            if 'else' in schema: validate_schema(value,schema['else'],path)
        else:
            if 'then' in schema: validate_schema(value,schema['then'],path)
