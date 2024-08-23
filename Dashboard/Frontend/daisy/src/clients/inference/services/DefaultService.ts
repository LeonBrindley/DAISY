/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { Body_upload_model_upload_model_post } from '../models/Body_upload_model_upload_model_post';
import type { InferenceResultSchemaOut } from '../models/InferenceResultSchemaOut';
import type { VisionModelSchemaOut } from '../models/VisionModelSchemaOut';
import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';
export class DefaultService {
    /**
     * Read Root
     * @returns any Successful Response
     * @throws ApiError
     */
    public static readRootGet(): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/',
        });
    }
    /**
     * Get All Inference Results
     * @returns InferenceResultSchemaOut Successful Response
     * @throws ApiError
     */
    public static getAllInferenceResultsInferenceResultsGet(): CancelablePromise<Array<InferenceResultSchemaOut>> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/inference_results',
        });
    }
    /**
     * Get All Models
     * @returns VisionModelSchemaOut Successful Response
     * @throws ApiError
     */
    public static getAllModelsModelsGet(): CancelablePromise<Array<VisionModelSchemaOut>> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/models',
        });
    }
    /**
     * Delete Model
     * @param modelId
     * @returns void
     * @throws ApiError
     */
    public static deleteModelModelsModelIdDelete(
        modelId: string,
    ): CancelablePromise<void> {
        return __request(OpenAPI, {
            method: 'DELETE',
            url: '/models/{model_id}',
            path: {
                'model_id': modelId,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Upload Model
     * @param formData
     * @param modelName
     * @returns VisionModelSchemaOut Successful Response
     * @throws ApiError
     */
    public static uploadModelUploadModelPost(
        formData: Body_upload_model_upload_model_post,
        modelName?: string,
    ): CancelablePromise<VisionModelSchemaOut> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/upload_model',
            query: {
                'model_name': modelName,
            },
            formData: formData,
            mediaType: 'multipart/form-data',
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Predict All
     * @returns any Successful Response
     * @throws ApiError
     */
    public static predictAllPredictAllPost(): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/predict_all',
        });
    }
    /**
     * Delete All Predictions
     * @returns any Successful Response
     * @throws ApiError
     */
    public static deleteAllPredictionsDeleteAllPredictionsDelete(): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'DELETE',
            url: '/delete_all_predictions',
        });
    }
    /**
     * Predict
     * @param imgUrl
     * @param imgId
     * @param modelId
     * @returns any Successful Response
     * @throws ApiError
     */
    public static predictPredictImgUrlPost(
        imgUrl: string,
        imgId: string,
        modelId: string,
    ): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/predict/{img_url}',
            path: {
                'img_url': imgUrl,
            },
            query: {
                'img_id': imgId,
                'model_id': modelId,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
}
