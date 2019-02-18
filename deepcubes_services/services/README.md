# Services, scripts and experiment section

Just several examples of usage

## Vera Live Dialog API

### /train

`GET` or `POST` query with `config` field as json string representaion:
```
{
	"lang": string,  # now just "rus" or "eng"
	"not_understand_label": string,  # any string that corresponds to `not understand` label
	"labels_settings": [
		{
			"label": string,  # unique label name
			"patterns": [string],  # list with regexps, example: ["нет", "нет.*"]
			"generics": [string],  # list with generics, possible:
                                               # ["yes", "no", "repeat", "no"questions"]
			"intent_phrases": [string],  # list with intent phrases for ML
		},
		...
	]
}
```

Returns json string with `model_id` (`int`).

### /predict

`GET` or `POST` query with `model_id` (`int`) field (returned by `/train`) and `query` (`string`) field as user input text. Additionall `labels` (`[array representation]`) field can be specified, in this case model returns probabilities only for specified labels.

Returns collection of labels sorted decreasingly according probabilities.

```
[
	{
		"label": string,
		"proba": float
	},
	...
]
```


## Embedder service

### /get_vector


`GET` or `POST` query with `mode` (`string`) (`"rus"` or `"eng"`), `tokens` (`[string]`) fields.

Returns json string with `vector` field, list of floats with embedding vector components.
